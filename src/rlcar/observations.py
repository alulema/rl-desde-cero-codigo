"""Qué ve el agente. Cada bloque se activa por separado, para las ablaciones del capítulo 7.

Con la configuración por defecto es el vector de `RaceAgent.CollectObservations` + los rayos, con dos
diferencias: el error de rumbo se mide contra la línea central (no hay línea de carrera) y la directiva es
un solo número.
"""

from dataclasses import dataclass

import numpy as np

from rlcar.sensors import cast_rays, mlagents_format
from rlcar.state import DriveState


@dataclass(frozen=True)
class ObsConfig:
    n_rays: int = 9
    ray_fov_deg: float = 150.0
    ray_length: float = 70.0
    ray_format: str = "distance"  # "distance": 1 número por rayo; "mlagents": 3, como en Agentic Racing
    speeds: bool = True
    heading_error: bool = True
    lateral_offset: bool = True
    progress: bool = True
    # "linear": el progreso en la vuelta, de 0 a 1. "sincos": su seno y su coseno, sin el salto de 1 a 0 en
    # la meta (el ejercicio de código del capítulo 7).
    progress_encoding: str = "linear"
    # Ventanas (desde, hasta) en metros por delante, como las 3 de race07/08.
    curvature_windows: tuple[tuple[float, float], ...] = ((0.0, 22.0), (18.0, 45.0), (40.0, 75.0))
    directive: bool = True
    previous_steer: bool = False
    stack: int = 1  # cuántas observaciones seguidas ve la policy (la actual y las anteriores), capítulo 7
    # Bug plantado (capítulo 8): desde este índice, el vector llega relleno con ceros. Tiene el tamaño
    # correcto, así que nada falla: el agente simplemente no ve esa parte.
    zero_fill_from: int | None = None


def observation_names(config: ObsConfig) -> list[str]:
    """El nombre de cada componente del vector, en orden. Útil para depurar y para el libro."""
    names = []
    if config.n_rays:
        if config.ray_format == "mlagents":
            names += [f"ray{i}_{part}" for i in range(config.n_rays) for part in ("hit", "miss", "dist")]
        else:
            names += [f"ray{i}" for i in range(config.n_rays)]
    if config.speeds:
        names += ["forward_speed", "lateral_speed"]
    if config.heading_error:
        names.append("heading_error")
    if config.lateral_offset:
        names.append("lateral_offset")
    if config.progress:
        if config.progress_encoding == "sincos":
            names += ["progress_sin", "progress_cos"]
        else:
            names.append("progress")
    names += [f"curvature_{int(a)}_{int(b)}m" for a, b in config.curvature_windows]
    if config.directive:
        names.append("aggression")
    if config.previous_steer:
        names.append("previous_steer")
    if config.stack > 1:
        names = [f"{name}_t-{k}" if k else name for k in range(config.stack) for name in names]
    return names


# region: build_observation
def build_observation(state: DriveState, config: ObsConfig,
                      max_speed: float) -> np.ndarray:
    parts = []
    car, projection, track = state.car, state.projection, state.track
    if config.n_rays:
        rays = cast_rays(
            car.position, car.heading, track, projection.index,
            config.n_rays, config.ray_fov_deg, config.ray_length,
        )
        if config.ray_format == "mlagents":
            rays = mlagents_format(rays)
        parts.append(rays)
    if config.speeds:
        parts.append([car.forward_speed / max_speed,
                      car.lateral_speed / max_speed])
    if config.heading_error:
        parts.append([state.heading_error() / np.pi])
    if config.lateral_offset:
        lateral = projection.lateral / track.half_width
        parts.append([np.clip(lateral, -2, 2)])
    if config.progress:
        if config.progress_encoding == "sincos":
            angle = 2 * np.pi * projection.progress
            parts.append([np.sin(angle), np.cos(angle)])
        else:
            parts.append([projection.progress])
    for start, end in config.curvature_windows:
        turn = track.turn_deg(projection.index, start, end)
        parts.append([np.clip(turn / 90, -1, 1)])
    if config.directive:
        parts.append([state.aggression])
    if config.previous_steer:
        last = state.last_action
        parts.append([0.0 if last is None else last[0]])
    observation = np.concatenate(
        [np.asarray(p, dtype=float) for p in parts]).astype(np.float32)
    if config.zero_fill_from is not None:
        observation[config.zero_fill_from:] = 0.0
    return observation
# endregion: build_observation
