"""Cuándo termina un episodio. Mismas reglas y mismo orden que `RaceAgent.OnActionReceived`.

Terminar también es una recompensa (capítulo 6): lo que vale el futuro que se pierde depende de qué paga
la recompensa por paso. Por eso cada regla se puede apagar o mover.
"""

from dataclasses import dataclass

from rlcar.state import DriveState


@dataclass(frozen=True)
class TerminationConfig:
    off_track_margin: float = 2.0  # metros fuera del borde
    stuck_speed: float = 0.5  # m/s
    stuck_seconds: float | None = 8.0  # None = estar detenido nunca termina el episodio
    wrong_way_speed: float = 7.5  # m/s hacia atrás (en el original: 0.15 m por paso de física de 0.02 s)
    wrong_way_seconds: float | None = 4.5
    wrong_way_arm_metres: float | None = 15.0  # solo cuenta después de avanzar esto; None = siempre (race04)
    stall_seconds: float | None = 5.0  # ventana del chequeo de progreso
    stall_min_metres: float = 8.0  # avance mínimo por ventana
    stall_grace_seconds: float = 1.2  # el original no lo aplica antes de 60 pasos de física
    lap_fraction: float = 0.99  # avance que cuenta como vuelta completa
    time_limit_seconds: float = 60.0  # corta el episodio (truncated, no terminated)


class TerminationTracker:
    """Los temporizadores de un episodio. Se crea uno nuevo en cada reset."""

    def __init__(self, config: TerminationConfig):
        self.config = config
        self.stuck_armed = False  # solo cuenta después de que el auto arrancó
        self.stuck_timer = 0.0
        self.wrong_way_timer = 0.0
        self.stall_timer = 0.0
        self.stall_start_metres = 0.0

    @property
    def stuck_active(self) -> bool:
        return self.stuck_armed and self.stuck_timer > 0

    # region: termination_update
    def update(self, state: DriveState, forward_metres: float, dt: float) -> str | None:
        """Actualiza los temporizadores con un paso de física y devuelve el motivo de fin, si lo hay.

        Suma `forward_metres` a `state.lap_metres` al final, como el original (que revisa el
        contrasentido y el estancamiento con el avance de antes de este paso).
        """
        c = self.config
        speed = state.car.forward_speed

        if abs(state.projection.lateral) > state.track.half_width + c.off_track_margin:
            return "off_track"

        if speed > c.stuck_speed:
            self.stuck_armed = True
        if self.stuck_armed and abs(speed) < c.stuck_speed:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0
        if c.stuck_seconds is not None and self.stuck_timer > c.stuck_seconds:
            return "stuck"

        backwards = forward_metres < -c.wrong_way_speed * dt
        armed = c.wrong_way_arm_metres is None or state.lap_metres > c.wrong_way_arm_metres
        if backwards and armed:
            self.wrong_way_timer += dt
        else:
            self.wrong_way_timer = 0.0
        if c.wrong_way_seconds is not None and self.wrong_way_timer > c.wrong_way_seconds:
            return "wrong_way"

        if c.stall_seconds is not None:
            self.stall_timer += dt
            if self.stall_timer >= c.stall_seconds:
                covered = state.lap_metres - self.stall_start_metres
                if state.time > c.stall_grace_seconds and covered < c.stall_min_metres:
                    return "stall"
                self.stall_start_metres = state.lap_metres
                self.stall_timer = 0.0

        state.lap_metres += forward_metres
        if state.lap_metres >= c.lap_fraction * state.track.length:
            return "lap"
        return None
    # endregion: termination_update
