"""Chequeos de depuración (capítulo 8): la secuencia que Agentic Racing habría querido seguir desde el
principio, en pasos que no necesitan entrenar.

Cada chequeo mide algo concreto y devuelve un diccionario con lo medido, `flag` (True = hay que mirar
esto) y `finding`, una frase con el hallazgo. Ninguno sabe qué bug buscar: comparan contra lo que debería
pasar en cualquier entorno de carreras sano.

1. `observation_check`: ¿la observación cambia? Un valor constante en cientos de pasos no le dice nada al
   agente (las observaciones rellenas con ceros de Agentic Racing).
2. `steering_check`: ¿la acción tiene el efecto que creemos? Girar no debería frenar ni acelerar el auto
   (el agarre que borraba la velocidad lateral).
3. `reward_check`: ¿la recompensa ordena bien los casos conocidos? Detenerse o no hacer nada no debería
   pagar más que la vuelta de la línea base (el atajo de la cuarta corrida).
4. `termination_check`: ¿alcanza el tiempo para dar la vuelta? (el límite de 80 s con vueltas de 95 s).
5. `baseline_check`: ¿la línea base da la vuelta, en varios episodios, y vuelve al centro desde un
   desvío conocido? (el signo invertido del controlador de Agentic Racing).
"""

from collections import Counter
import numpy as np

from rlcar.env import EnvConfig, RLCarEnv
from rlcar.evaluate import Policy, evaluate, summarize
from rlcar.physics import CarParams, CarState
from rlcar.physics import step as physics_step


class RandomPolicy:
    """Acciones uniformes en todo el rango: la prueba más barata de que la tubería corre."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, observation, env):
        return self.rng.uniform(env.action_space.low, env.action_space.high)


class ConstantPolicy:
    """Siempre la misma acción, por ejemplo frenar a fondo desde el arranque."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def reset(self) -> None:
        pass

    def act(self, observation, env):
        return self.action


# region: observation_check
def observation_check(config: EnvConfig, policies: list[Policy], seeds) -> dict:
    """Junta las observaciones de varios episodios y busca componentes que nunca cambian.

    Un valor constante no siempre es un bug: la agresión no cambia dentro de un episodio, y con pocos
    episodios puede salir igual en todos. Es una marca para ir a mirar, no un veredicto."""
    observations = []
    for policy in policies:
        for seed in seeds:
            env = RLCarEnv(config)
            observation, _ = env.reset(seed=seed)
            policy.reset()
            observations.append(observation)
            while True:
                observation, _, terminated, truncated, _ = env.step(policy.act(observation, env))
                observations.append(observation)
                if terminated or truncated:
                    break
    observations = np.array(observations)
    names = env.observation_names
    std = observations.std(axis=0)
    constant = [name for name, s in zip(names, std) if s < 1e-6]
    return {
        "samples": len(observations),
        "constant": constant,
        "constant_values": {name: float(observations[0, names.index(name)]) for name in constant},
        "std": {name: float(s) for name, s in zip(names, std)},
        "flag": bool(constant),
        "finding": (f"{len(constant)} de {len(names)} valores no cambian nunca: {', '.join(constant)}"
                    if constant else f"los {len(names)} valores cambian"),
    }
# endregion: observation_check


# region: steering_check
def steering_check(car: CarParams, speed: float = 20.0,
                   seconds: float = 1.0, dt: float = 0.02) -> dict:
    """Solo la física: el auto a `speed` sin acelerador ni freno, recto y
    con el volante a fondo.

    En los dos casos pierde la misma velocidad por el rozamiento; si
    girando pierde (o gana) bastante más, el giro está cambiando la rapidez
    del auto.
    """
    final = {}
    for name, steer in (("straight", 0.0), ("full_steer", 1.0)):
        state = CarState(0.0, 0.0, 0.0, speed, 0.0)
        action = np.array([steer, 0.0, 0.0])
        for _ in range(round(seconds / dt)):
            state = physics_step(state, action, dt, car)
        final[name] = float(np.hypot(state.vx, state.vy))
    straight, turning = final["straight"], final["full_steer"]
    change = (turning - straight) / straight
    return {
        "speed": speed, "seconds": seconds,
        "speed_straight": straight, "speed_full_steer": turning,
        "turn_speed_change": change,
        "flag": abs(change) > 0.05,
        "finding": (f"en {seconds:.0f} s, girando a fondo termina a "
                    f"{turning:.1f} m/s y recto a {straight:.1f} m/s "
                    f"({change:+.0%})"),
    }
# endregion: steering_check


# region: reward_check
def reward_check(config: EnvConfig, baseline: Policy, seeds) -> dict:
    """El return de la línea base contra el de dos comportamientos que no deberían pagar: frenar a fondo
    desde el arranque y no tocar nada."""
    cases = {"baseline": baseline, "brake": ConstantPolicy([0.0, 0.0, 1.0]), "idle": ConstantPolicy([0.0, 0.0, 0.0])}
    returns = {name: float(np.mean([r.episode_return for r in evaluate(config, policy, seeds)]))
               for name, policy in cases.items()}
    better = [name for name in ("brake", "idle") if returns[name] >= returns["baseline"]]
    return {
        "returns": returns,
        "flag": bool(better),
        "finding": ("la línea base cobra " + f"{returns['baseline']:.2f}; frenar, {returns['brake']:.2f}; "
                    f"no hacer nada, {returns['idle']:.2f}"
                    + (f". Paga más: {', '.join(better)}" if better else "")),
    }
# endregion: reward_check


# region: termination_check
def termination_check(config: EnvConfig, baseline: Policy, seeds) -> dict:
    """¿Alcanza el límite de tiempo? La velocidad media que hace falta para
    la vuelta en ese límite, contra la que lleva la línea base, y cuántos
    de sus episodios corta el reloj."""
    records = evaluate(config, baseline, seeds)
    env = RLCarEnv(config)
    env.reset(seed=seeds[0])
    limit = config.termination.time_limit_seconds
    lap = config.termination.lap_fraction * env.track.length
    needed = lap / limit
    baseline_speed = float(np.mean([r.mean_speed for r in records]))
    cut = [r for r in records if r.end_reason == "time_limit"]
    reasons = Counter(r.end_reason for r in records)
    return {
        "time_limit_seconds": limit,
        "track_length": env.track.length,
        "speed_needed": needed,
        "baseline_mean_speed": baseline_speed,
        "baseline_end_reasons": dict(reasons),
        "baseline_time_limit_fraction": (
            float(np.mean([r.lap_fraction for r in cut])) if cut else None),
        "flag": needed > baseline_speed or bool(cut),
        "finding": (f"para la vuelta en {limit:.0f} s hace falta una media "
                    f"de {needed:.1f} m/s; la línea base lleva "
                    f"{baseline_speed:.1f} m/s y el reloj corta {len(cut)} "
                    f"de {len(records)} episodios"),
    }
# endregion: termination_check


def straight_index(env: RLCarEnv, metres: float = 40.0) -> int:
    """El punto de la pista con menos curva en los próximos `metres`."""
    return min(range(env.track.n), key=lambda i: abs(env.track.turn_deg(i, 0.0, metres)))


# region: baseline_check
def baseline_check(config: EnvConfig, baseline: Policy, seeds, offset: float = 4.0, speed: float = 15.0,
                   seconds: float = 3.0) -> dict:
    """Dos preguntas. ¿Da la vuelta en varios episodios, no en uno? Y un caso conocido: en una recta, a
    `offset` metros del centro, alineado y a `speed`, ¿vuelve al centro en `seconds`?"""
    summary = summarize(evaluate(config, baseline, seeds))
    env = RLCarEnv(config)
    env.reset(seed=seeds[0])
    index = straight_index(env)
    heading = env.track.segment_headings[index]
    left = np.array([-np.sin(heading), np.cos(heading)])
    remaining = []
    for side in (1.0, -1.0):
        env.reset(seed=seeds[0])
        baseline.reset()
        position = env.track.points[index] + side * offset * left
        env.state.car = CarState(position[0], position[1], heading, speed * np.cos(heading), speed * np.sin(heading))
        env.state.projection = env.track.project(env.state.car.position)
        for _ in range(round(seconds / (config.physics_dt * config.decision_period))):
            _, _, terminated, truncated, _ = env.step(baseline.act(None, env))
            if terminated or truncated:
                break
        remaining.append(abs(env.state.projection.lateral))
    laps = summary["completed_laps"] / summary["episodes"]
    return {
        "summary": summary,
        "offset": offset, "speed": speed, "seconds": seconds,
        "remaining_offset": float(max(remaining)),
        "flag": laps < 0.5 or max(remaining) > 1.0,
        "finding": (f"da la vuelta en {summary['completed_laps']} de {summary['episodes']} episodios "
                    f"(fracción media {summary['lap_fraction_mean']:.2f}); desde {offset:.0f} m del centro, "
                    f"a los {seconds:.0f} s sigue a {max(remaining):.1f} m"),
    }
# endregion: baseline_check


def run_all(config: EnvConfig, baseline: Policy, seeds) -> dict:
    """Los cinco chequeos, en el orden de la secuencia del capítulo 8."""
    return {
        "observations": observation_check(config, [RandomPolicy(), baseline], seeds),
        "steering": steering_check(config.car),
        "reward": reward_check(config, baseline, seeds),
        "termination": termination_check(config, baseline, seeds),
        "baseline": baseline_check(config, baseline, seeds),
    }
