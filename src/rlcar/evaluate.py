"""Evaluador independiente de la recompensa (capítulo 8).

La recompensa dice cuánto le pagamos al agente; no dice si maneja bien. El evaluador mide lo que nos
importa: fracción de vuelta, vueltas completas, tiempo por vuelta y por qué terminó cada episodio. El
return se reporta aparte, solo para compararlo.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from rlcar.env import EnvConfig, RLCarEnv
from rlcar.expert import Expert


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: np.ndarray, env: RLCarEnv) -> np.ndarray: ...


class ExpertPolicy:
    """El experto como policy: lee el estado privilegiado del entorno, no la observación."""

    def __init__(self, memory: bool = True, sign_bug: bool = False):
        self.expert = Expert(memory=memory, sign_bug=sign_bug)

    def reset(self) -> None:
        self.expert.reset()

    def act(self, observation, env):
        return self.expert.act(env.state)


class FunctionPolicy:
    """Cualquier función de observación a acción, por ejemplo `lambda o: model.predict(o, deterministic=True)[0]`."""

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self.function = function

    def reset(self) -> None:
        pass

    def act(self, observation, env):
        return self.function(observation)


@dataclass
class EpisodeRecord:
    seed: int
    end_reason: str
    lap_fraction: float
    completed_lap: bool
    duration: float  # segundos
    lap_time: float | None  # solo si completó la vuelta
    mean_speed: float
    mean_throttle: float
    mean_brake: float
    mean_abs_steer: float
    episode_return: float
    discounted_return: float  # con el γ del PPO del libro: lo que PPO optimiza
    reward_terms: dict[str, float] = field(default_factory=dict)  # cada término sumado sobre el episodio


# region: run_episode
def run_episode(config: EnvConfig, policy: Policy, seed: int, log_every: int | None = None, gamma: float = 0.995):
    """Corre un episodio. Devuelve su registro y, si `log_every` no es None, una fila cada `log_every`
    decisiones con la posición, la velocidad y la acción. `gamma` solo se usa para el return descontado."""
    env = RLCarEnv(config)
    observation, _ = env.reset(seed=seed)
    policy.reset()
    actions, speeds, rows = [], [], []
    totals: Counter = Counter()
    discounted, weight = 0.0, 1.0
    while True:
        action = np.clip(np.asarray(policy.act(observation, env), dtype=float), env.action_space.low,
                         env.action_space.high)
        observation, reward, terminated, truncated, info = env.step(action)
        totals.update(info["reward_terms"])
        discounted += weight * reward
        weight *= gamma
        actions.append(action)
        speeds.append(env.state.car.forward_speed)
        if log_every and len(actions) % log_every == 0:
            car = env.state.car
            rows.append({"seed": seed, "time": round(env.state.time, 3), "x": car.x, "y": car.y,
                         "speed": car.forward_speed, "lap_fraction": env.state.lap_fraction,
                         "steer": action[0], "throttle": action[1], "brake": action[2]})
        if terminated or truncated:
            break
    actions = np.array(actions)
    completed = info["end_reason"] == "lap"
    record = EpisodeRecord(
        seed=seed,
        end_reason=info["end_reason"],
        lap_fraction=float(min(1.0, max(0.0, env.state.lap_fraction))),
        completed_lap=completed,
        duration=env.state.time,
        lap_time=env.state.time if completed else None,
        mean_speed=float(np.mean(speeds)),
        mean_throttle=float(np.mean(actions[:, 1])),
        mean_brake=float(np.mean(actions[:, 2])),
        mean_abs_steer=float(np.mean(np.abs(actions[:, 0]))),
        episode_return=float(sum(totals.values())),
        discounted_return=float(discounted),
        reward_terms=dict(totals),
    )
    return record, rows
# endregion: run_episode


def evaluate(config: EnvConfig, policy: Policy, seeds) -> list[EpisodeRecord]:
    return [run_episode(config, policy, seed)[0] for seed in seeds]


def summarize(records: list[EpisodeRecord]) -> dict:
    """Medias y dispersión de un conjunto de episodios. La dispersión importa tanto como la media: un
    promedio puede no describir a ningún episodio (los 5.3 m/s de la bitácora)."""
    lap_times = [r.lap_time for r in records if r.lap_time is not None]
    fractions = np.array([r.lap_fraction for r in records])
    return {
        "episodes": len(records),
        "lap_fraction_mean": float(fractions.mean()),
        "lap_fraction_std": float(fractions.std()),
        "lap_fraction_min": float(fractions.min()),
        "completed_laps": sum(r.completed_lap for r in records),
        "lap_time_mean": float(np.mean(lap_times)) if lap_times else None,
        "end_reasons": dict(Counter(r.end_reason for r in records)),
        "mean_speed": float(np.mean([r.mean_speed for r in records])),
        "mean_brake": float(np.mean([r.mean_brake for r in records])),
        "return_mean": float(np.mean([r.episode_return for r in records])),
        "discounted_return_mean": float(np.mean([r.discounted_return for r in records])),
    }
