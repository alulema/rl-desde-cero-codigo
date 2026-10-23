"""Rayos contra los bordes de la pista: el equivalente de `RayPerceptionSensorComponent3D`.

En Agentic Racing eran 9 rayos repartidos en 150° (75° a cada lado), de 70 m. Aquí cada rayo devuelve la
distancia al primer borde que toca, dividida por el largo del rayo: 1.0 si no toca nada.
"""

import numpy as np

from rlcar.track import Track


def ray_angles(n_rays: int, fov_deg: float) -> np.ndarray:
    """Ángulos de los rayos respecto a la trompa, de derecha a izquierda."""
    if n_rays == 1:
        return np.zeros(1)
    half = np.radians(fov_deg) / 2
    return np.linspace(-half, half, n_rays)


def edge_segments(track: Track, center_index: int, reach_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Segmentos de los dos bordes cerca del auto, para no probar los rayos contra toda la pista."""
    window = int(np.ceil(reach_m / track.spacing)) + 2
    indices = (center_index + np.arange(-window, window + 1)) % track.n
    following = (indices + 1) % track.n
    starts = np.vstack([track.left_edge[indices], track.right_edge[indices]])
    ends = np.vstack([track.left_edge[following], track.right_edge[following]])
    return starts, ends


# region: cast_rays
def cast_rays(
    position: np.ndarray,
    heading: float,
    track: Track,
    center_index: int,
    n_rays: int = 9,
    fov_deg: float = 150.0,
    length: float = 70.0,
) -> np.ndarray:
    """Distancia normalizada (0..1) de cada rayo al borde más cercano en su dirección."""
    angles = heading + ray_angles(n_rays, fov_deg)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (rayos, 2)
    starts, ends = edge_segments(track, center_index, length)
    segments = ends - starts  # (segmentos, 2)

    # Rayo: position + t·d, t ≥ 0. Segmento: start + u·s, 0 ≤ u ≤ 1. Se resuelve el cruce para todos los
    # pares a la vez con productos cruzados en 2D.
    d = directions[:, None, :]
    s = segments[None, :, :]
    offset = (starts - position)[None, :, :]
    denominator = d[..., 0] * s[..., 1] - d[..., 1] * s[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset[..., 0] * s[..., 1] - offset[..., 1] * s[..., 0]) / denominator
        u = (offset[..., 0] * d[..., 1] - offset[..., 1] * d[..., 0]) / denominator
    hits = (np.abs(denominator) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
    distances = np.where(hits, t, np.inf).min(axis=1)
    return np.minimum(distances, length) / length
# endregion: cast_rays


def mlagents_format(normalized: np.ndarray) -> np.ndarray:
    """Los 3 números por rayo de ML-Agents con una sola etiqueta: [tocó borde, no tocó nada, fracción]."""
    hit = normalized < 1.0
    return np.stack([hit, ~hit, normalized], axis=1).astype(float).ravel()
