"""El estado privilegiado de un episodio: lo que sabe el entorno, no lo que ve el agente.

Lo usan la observación, la recompensa, el experto y el evaluador. El agente nunca lo recibe directamente:
solo ve la observación que se construye a partir de él.
"""

from dataclasses import dataclass

import numpy as np

from rlcar.physics import CarState
from rlcar.track import Projection, Track


@dataclass
class DriveState:
    track: Track
    car: CarState
    projection: Projection
    aggression: float
    time: float = 0.0  # segundos desde el inicio del episodio
    lap_metres: float = 0.0  # avance neto en la vuelta desde el spawn
    last_action: np.ndarray | None = None  # la acción de la decisión anterior

    @property
    def lap_fraction(self) -> float:
        return self.lap_metres / self.track.length

    def heading_error(self) -> float:
        """Ángulo con signo de la trompa a la dirección de la pista, en radianes (+ = la pista va a la
        izquierda de la trompa)."""
        tangent = self.projection.tangent
        forward = self.car.forward
        return float(np.arctan2(forward[0] * tangent[1] - forward[1] * tangent[0], np.dot(forward, tangent)))
