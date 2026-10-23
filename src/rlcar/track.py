"""La pista: una línea central cerrada, muestreada cada ~1 m, con un ancho fijo.

Es la versión en Python de `TrackProgress` y de la parte de `TrackGenerator` que importa para el libro:
dónde está el auto respecto a la pista (progreso, desplazamiento lateral) y cuánto gira la pista por
delante (la curvatura que el agente necesita para frenar a tiempo).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    """Dónde está un punto respecto a la línea central."""

    index: int  # muestra más cercana de la línea central
    progress: float  # 0..1 alrededor de la vuelta, desde la línea de meta
    lateral: float  # metros desde la línea central, + = a la izquierda del sentido de marcha
    tangent: np.ndarray  # dirección unitaria de la pista en ese punto


class Track:
    """Circuito cerrado. `points` no repite el primer punto al final."""

    def __init__(self, points: np.ndarray, width: float):
        self.points = np.asarray(points, dtype=float)
        self.width = float(width)
        self.half_width = self.width / 2
        self.n = len(self.points)

        segments = np.roll(self.points, -1, axis=0) - self.points
        self.segment_lengths = np.linalg.norm(segments, axis=1)
        self.length = float(self.segment_lengths.sum())
        self.spacing = self.length / self.n
        # Rumbo del segmento i (de la muestra i a la i+1), en radianes.
        self.segment_headings = np.arctan2(segments[:, 1], segments[:, 0])

        # Tangente centrada en cada muestra, como TrackProgress (c - a).
        centred = np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)
        self.tangents = centred / np.linalg.norm(centred, axis=1, keepdims=True)
        normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)  # a la izquierda
        self.left_edge = self.points + normals * self.half_width
        self.right_edge = self.points - normals * self.half_width

    # region: projection
    def project(self, position: np.ndarray, hint: int | None = None, window: int = 60) -> Projection:
        """Proyecta un punto sobre la línea central.

        Con `hint` (la muestra más cercana del paso anterior) solo busca en una ventana alrededor, como
        `TrackProgress.Update`. Sin `hint` recorre toda la pista, como `TrackProgress.Reset`.
        """
        if hint is None:
            candidates = np.arange(self.n)
        else:
            candidates = (hint + np.arange(-window, window + 1)) % self.n
        distances = np.linalg.norm(self.points[candidates] - position, axis=1)
        index = int(candidates[np.argmin(distances)])

        # Fracción a lo largo del segmento [index, index+1], para que el progreso sea continuo.
        start = self.points[index]
        segment = self.points[(index + 1) % self.n] - start
        fraction = np.clip(np.dot(position - start, segment) / np.dot(segment, segment), 0.0, 1.0)
        progress = ((index + fraction) % self.n) / self.n

        tangent = self.tangents[index]
        left = np.array([-tangent[1], tangent[0]])
        lateral = float(np.dot(position - start, left))
        return Projection(index=index, progress=float(progress), lateral=lateral, tangent=tangent)
    # endregion: projection

    def forward_metres(self, previous_progress: float, progress: float) -> float:
        """Avance en metros entre dos progresos, sin saltos en la meta (0.99 → 0.01 es avanzar)."""
        delta = progress - previous_progress
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0
        return delta * self.length

    def turn_deg(self, index: int, from_m: float, to_m: float) -> float:
        """Cuánto gira la línea central entre `from_m` y `to_m` metros por delante de `index`.

        Con signo: + = la pista dobla a la izquierda. Es `CenterlineTurnDeg` de RaceAgent.
        """
        i0 = (index + round(from_m / self.spacing)) % self.n
        i1 = (index + round(to_m / self.spacing)) % self.n
        return float(np.degrees(wrap_angle(self.segment_headings[i1] - self.segment_headings[i0])))

    def corner_radii(self, stencil_m: float = 6.0) -> np.ndarray:
        """Radio de la circunferencia que pasa por las muestras i−k, i, i+k (k ≈ stencil_m)."""
        k = max(1, round(stencil_m / self.spacing))
        a = np.roll(self.points, k, axis=0)
        b = self.points
        c = np.roll(self.points, -k, axis=0)
        ab = np.linalg.norm(b - a, axis=1)
        bc = np.linalg.norm(c - b, axis=1)
        ca = np.linalg.norm(a - c, axis=1)
        cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        with np.errstate(divide="ignore"):
            return np.where(cross > 1e-9, ab * bc * ca / (2 * cross), np.inf)

    def min_corner_radius(self) -> float:
        return float(self.corner_radii().min())


def wrap_angle(angle):
    """Lleva un ángulo (o un arreglo) a [−π, π)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Remuestrea una polilínea cerrada a distancias iguales a lo largo de su longitud."""
    closed = np.vstack([points, points[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    n = max(8, round(arc[-1] / spacing))
    targets = np.linspace(0.0, arc[-1], n, endpoint=False)
    x = np.interp(targets, arc, closed[:, 0])
    y = np.interp(targets, arc, closed[:, 1])
    return np.stack([x, y], axis=1)


# region: fixed_circuit
def fixed_circuit(
    half_length: float = 120.0,
    half_height: float = 60.0,
    corner_radius: float = 40.0,
    width: float = 12.0,
    spacing: float = 1.0,
) -> Track:
    """Rectángulo de esquinas redondeadas, recorrido en sentido antihorario.

    Es el circuito fijo de Agentic Racing (`TrackParams.FixedRoundedRect`) a escala. La meta está en el
    centro de la recta inferior y el auto sale hacia +x.
    """
    straight_x = half_length - corner_radius
    straight_y = half_height - corner_radius
    pieces = []
    # Cada esquina: centro del arco y ángulo inicial, recorriendo en sentido antihorario.
    corners = [
        ((straight_x, -straight_y), -np.pi / 2),
        ((straight_x, straight_y), 0.0),
        ((-straight_x, straight_y), np.pi / 2),
        ((-straight_x, -straight_y), np.pi),
    ]
    pieces.append(np.array([[0.0, -half_height], [straight_x, -half_height]]))
    for (cx, cy), start in corners:
        angles = np.linspace(start, start + np.pi / 2, 64)
        pieces.append(np.stack([cx + corner_radius * np.cos(angles), cy + corner_radius * np.sin(angles)], 1))
    pieces.append(np.array([[-straight_x, -half_height], [0.0, -half_height]]))
    outline = np.vstack(pieces)
    # Quita puntos repetidos en las uniones antes de remuestrear.
    keep = np.r_[True, np.linalg.norm(np.diff(outline, axis=0), axis=1) > 1e-9]
    outline = outline[keep][:-1]
    return Track(resample_closed(outline, spacing), width)
# endregion: fixed_circuit


def circuit_from_segments(segments: list[tuple], width: float = 12.0, spacing: float = 1.0) -> Track:
    """Arma un circuito como un dibujo de tortuga: ("straight", metros) o ("arc", radio, grados).

    Los grados positivos giran a la izquierda. La lista tiene que cerrar el lazo: girar 360° en total y
    volver al punto de partida. Se arranca en el origen, mirando hacia +x.
    """
    position, heading = np.zeros(2), 0.0
    points = [position.copy()]
    for segment in segments:
        if segment[0] == "straight":
            n = max(2, round(segment[1] / spacing))
            for d in np.linspace(0, segment[1], n + 1)[1:]:
                points.append(position + d * np.array([np.cos(heading), np.sin(heading)]))
            position = points[-1].copy()
        else:
            _, radius, degrees = segment
            sign = np.sign(degrees)
            center = position + sign * radius * np.array([-np.sin(heading), np.cos(heading)])
            start = np.arctan2(*(position - center)[::-1])
            n = max(2, round(abs(np.radians(degrees)) * radius / spacing))
            for a in np.linspace(0, np.radians(degrees), n + 1)[1:]:
                points.append(center + radius * np.array([np.cos(start + a), np.sin(start + a)]))
            position, heading = points[-1].copy(), heading + np.radians(degrees)
    if np.linalg.norm(points[-1] - points[0]) > spacing:
        raise ValueError(f"los segmentos no cierran el lazo: terminan a {np.linalg.norm(points[-1]):.1f} m del inicio")
    return Track(resample_closed(np.array(points[:-1]), spacing), width)


# region: technical_circuit
def technical_circuit(straight: float = 200.0, hairpin_radius: float = 15.0, width: float = 12.0) -> Track:
    """Óvalo tipo estadio: dos rectas largas unidas por horquillas de 180°.

    En la recta el auto llega a mucha velocidad, y en una horquilla no se puede abrir la trazada como en
    una curva de 90°: hay que frenar antes. La meta está al inicio de la recta inferior.
    """
    segments = [("straight", straight), ("arc", hairpin_radius, 180), ("straight", straight),
                ("arc", hairpin_radius, 180)]
    return circuit_from_segments(segments, width)
# endregion: technical_circuit


def random_circuit(
    seed: int,
    base_radius: float = 100.0,
    control_points: tuple[int, int] = (12, 16),
    harmonics: tuple[int, int] = (1, 2),
    harmonic_freq: tuple[int, int] = (2, 3),
    harmonic_amp: tuple[float, float] = (0.08, 0.16),
    radial_jitter: float = 0.035,
    min_corner_radius: float | None = 30.0,
    width: float = 12.0,
    spacing: float = 1.0,
    max_attempts: int = 120,
) -> Track:
    """Circuito aleatorio con semilla, en el estilo de `TrackGenerator`.

    Un radio base modulado por unas pocas ondas (que crean rectas y curvas) más un poco de ruido, en
    coordenadas polares: con radio siempre positivo, la curva nunca se cruza consigo misma. Se suaviza
    con un spline de Catmull-Rom.

    Si `min_corner_radius` no es None, se descartan los intentos con curvas más cerradas que ese radio.
    Con `min_corner_radius=None` no se filtra nada: así se generan las curvas muy cerradas de los capítulos 6 y 7.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        n = rng.integers(control_points[0], control_points[1] + 1)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        modulation = np.zeros(n)
        for _ in range(rng.integers(harmonics[0], harmonics[1] + 1)):
            freq = rng.integers(harmonic_freq[0], harmonic_freq[1] + 1)
            amp = rng.uniform(*harmonic_amp)
            modulation += amp * np.sin(freq * angles + rng.uniform(0, 2 * np.pi))
        modulation += rng.uniform(-radial_jitter, radial_jitter, n)
        radii = base_radius * np.clip(1 + modulation, 0.6, 1.5)
        control = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        track = Track(resample_closed(catmull_rom_closed(control), spacing), width)
        if min_corner_radius is None or track.min_corner_radius() >= min_corner_radius:
            return track
    raise ValueError(f"no se generó una pista válida con la semilla {seed} en {max_attempts} intentos")


def catmull_rom_closed(control: np.ndarray, samples_per_segment: int = 40) -> np.ndarray:
    """Spline de Catmull-Rom (uniforme) que pasa por todos los puntos de control de un lazo cerrado."""
    p0 = np.roll(control, 1, axis=0)
    p1 = control
    p2 = np.roll(control, -1, axis=0)
    p3 = np.roll(control, -2, axis=0)
    t = np.linspace(0, 1, samples_per_segment, endpoint=False)[None, :, None]
    curve = 0.5 * (
        2 * p1[:, None]
        + (-p0 + p2)[:, None] * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None] * t**2
        + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None] * t**3
    )
    return curve.reshape(-1, 2)
