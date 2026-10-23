"""La recompensa, término por término.

Cada término tiene su peso y su unidad (por metro, por segundo, por paso o por evento), igual que en
`RaceAgent.OnActionReceived`. Todos los pesos valen 0 por defecto: los presets de `rlcar.presets` arman las
recompensas de Agentic Racing. El entorno devuelve cada término por separado en `info["reward_terms"]`,
porque sin eso no se puede saber qué está pagando de verdad (capítulo 6).
"""

from dataclasses import dataclass

import numpy as np

from rlcar.directive import speed_scale
from rlcar.physics import clamp, lerp
from rlcar.state import DriveState


@dataclass(frozen=True)
class RewardConfig:
    # Por paso de física, proporcionales al avance o al tiempo.
    progress_per_metre: float = 0.0
    target_speed_per_sec: float = 0.0  # pico en la velocidad objetivo según la curva (race07/08)
    speed_per_sec: float = 0.0  # proporcional a v / max_speed (race02–04)
    line_follow_per_sec: float = 0.0  # alineado con la pista y cerca de la línea central
    slow_per_sec: float = 0.0  # castigo por ir muy por debajo del objetivo
    edge_per_sec: float = 0.0  # castigo por acercarse al borde
    edge_ramp: bool = True  # True: rampa desde edge_safe_frac (race06+); False: solo fuera de la pista (race04)
    edge_safe_frac: float = 0.55
    time_per_step: float = 0.0  # castigo plano por paso de física (race01–04)
    stuck_per_sec: float = 0.0  # castigo mientras está detenido (race05+)
    alive_per_sec: float = 0.0  # premio constante por seguir vivo (no existía: capítulo 4 y 6)
    # Por evento, al terminar el episodio. Se escriben en positivo y se restan.
    off_track: float = 0.0
    stuck: float = 0.0
    wrong_way: float = 0.0
    stall: float = 0.0
    lap_bonus: float = 0.0
    fast_lap_bonus: float = 0.0  # se multiplica por el tiempo que sobra del límite
    # Velocidad objetivo (TargetSpeed() de RaceAgent).
    straight_speed_frac: float = 0.42
    corner_speed_frac: float = 0.12
    directive_scales_target: bool = False  # la propuesta del capítulo 6: que la recompensa lea la directiva


# region: target_speed
def target_speed(state: DriveState, config: RewardConfig,
                 max_speed: float) -> float:
    """Velocidad que la recompensa considera ideal: alta en recta, baja
    antes de la curva más cerrada de los próximos 55 m."""
    index, track = state.projection.index, state.track
    turn = max(abs(track.turn_deg(index, 0, 30)),
               abs(track.turn_deg(index, 20, 55)))
    target = max_speed * lerp(config.straight_speed_frac,
                              config.corner_speed_frac,
                              clamp(turn / 55, 0, 1))
    if config.directive_scales_target:
        target *= speed_scale(state.aggression)
    return target
# endregion: target_speed


# region: step_terms
def step_terms(
    state: DriveState, forward_metres: float, stuck_active: bool, dt: float, config: RewardConfig, max_speed: float
) -> dict[str, float]:
    """Los términos de un paso de física. Solo aparecen los que tienen peso distinto de 0."""
    terms = {}
    car = state.car
    speed = car.forward_speed
    forward_fraction = clamp(speed / max_speed, 0, 1)
    abs_lateral = abs(state.projection.lateral)
    half_width = state.track.half_width

    if config.progress_per_metre:
        terms["progress"] = config.progress_per_metre * forward_metres

    if config.target_speed_per_sec or config.slow_per_sec:
        target = target_speed(state, config, max_speed)
        error = (speed - target) / max(1.0, target)  # con signo: < 0 es ir lento
        if config.target_speed_per_sec:
            terms["target_speed"] = config.target_speed_per_sec * clamp(1 - abs(error) * 1.3, 0, 1) * dt
        if config.slow_per_sec and error < -0.3:
            terms["slow"] = -config.slow_per_sec * (-error - 0.3) * dt

    if config.speed_per_sec:
        terms["speed"] = config.speed_per_sec * forward_fraction * dt

    if config.line_follow_per_sec and forward_fraction > 0.05:
        heading_error01 = abs(state.heading_error()) / np.pi
        line_distance01 = clamp(abs_lateral / half_width, 0, 1)
        follow = (1 - clamp(heading_error01 * 3, 0, 1)) * (1 - line_distance01)
        terms["line_follow"] = config.line_follow_per_sec * follow * forward_fraction * dt

    if config.edge_per_sec:
        if config.edge_ramp:
            safe = config.edge_safe_frac * half_width
            ramp = clamp((abs_lateral - safe) / (half_width - safe), 0, 1)
        else:
            ramp = 1.0 if abs_lateral > half_width else 0.0
        if ramp > 0:
            terms["edge"] = -config.edge_per_sec * ramp * dt

    if config.time_per_step:
        terms["time"] = -config.time_per_step
    if config.stuck_per_sec and stuck_active:
        terms["stuck"] = -config.stuck_per_sec * dt
    if config.alive_per_sec:
        terms["alive"] = config.alive_per_sec * dt
    return terms
# endregion: step_terms


# region: end_terms
def end_terms(reason: str, time_left_fraction: float,
              config: RewardConfig) -> dict[str, float]:
    """El término del evento que terminó el episodio (si tiene peso)."""
    if reason == "lap":
        bonus = (config.lap_bonus
                 + config.fast_lap_bonus * time_left_fraction)
        return {"lap": bonus} if bonus else {}
    penalty = {"off_track": config.off_track, "stuck": config.stuck,
               "wrong_way": config.wrong_way,
               "stall": config.stall}.get(reason, 0.0)
    return {f"end_{reason}": -penalty} if penalty else {}
# endregion: end_terms
