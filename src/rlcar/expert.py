"""El experto heurístico: una adaptación de `RaceAgent.Heuristic()` de Agentic Racing.

Lee el estado privilegiado (`env.unwrapped.state`), no la observación. Fija una velocidad objetivo según
la curva más cerrada de los próximos ~60 m, acelera o frena para alcanzarla, y apunta el volante a un
punto entre 12 y 36 m por delante en la línea central (persecución pura).

Hay dos variantes:
- `memory=True`: filtra el volante con el comando anterior y tiene una reversa de escape con
  temporizador, como el original. Su acción depende de la historia, no solo del estado.
- `memory=False`: sin filtro ni temporizadores. Es una función pura del estado.

Para DAgger, llama a `act` en cada paso, también cuando maneja la policy: así el filtro y los
temporizadores siguen la trayectoria real.

`sign_bug=True` es un bug plantado para el capítulo 8: la corrección hacia el centro con el signo
invertido. La línea base de Agentic Racing también tuvo un signo invertido en su primera evaluación.
"""

import numpy as np

from rlcar.directive import brake_margin, speed_scale
from rlcar.physics import clamp, lerp
from rlcar.state import DriveState
from rlcar.track import wrap_angle


class Expert:
    def __init__(self, memory: bool = True, decision_dt: float = 0.1, max_speed: float = 55.0,
                 sign_bug: bool = False):
        self.memory = memory
        self.sign_bug = sign_bug
        self.decision_dt = decision_dt
        self.max_speed = max_speed
        self.reset()

    def reset(self) -> None:
        self.steer_smooth = 0.0
        self.wall_jam_timer = 0.0
        self.escape_time_left = 0.0

    # region: expert_target_speed
    def target_speed(self, state: DriveState) -> float:
        """El giro más cerrado de la línea central en los próximos ~60 m fija la velocidad de entrada."""
        track, index = state.track, state.projection.index
        step = max(1, round(8 / track.spacing))
        scan = max(2, round(60 / track.spacing))
        sharpest = 0.0
        for k in range(0, scan, step):
            change = wrap_angle(track.segment_headings[(index + k + step) % track.n]
                                - track.segment_headings[(index + k) % track.n])
            sharpest = max(sharpest, abs(np.degrees(change)))
        base = lerp(self.max_speed * 0.40, self.max_speed * 0.10, clamp(sharpest / 45, 0, 1))
        return base * speed_scale(state.aggression)
    # endregion: expert_target_speed

    def aim_error(self, state: DriveState) -> float:
        """Grados que hay que girar la trompa para apuntar a un punto de la línea central entre 12 y 36 m
        por delante (más lejos cuanto más rápido): + = a la izquierda."""
        track, car, projection = state.track, state.car, state.projection
        lookahead_m = clamp(12 + max(0.0, car.forward_speed) * 0.9, 12, 36)
        aim = track.points[(projection.index + round(lookahead_m / track.spacing)) % track.n]
        to_aim = aim - car.position
        return float(np.degrees(wrap_angle(np.arctan2(to_aim[1], to_aim[0]) - car.heading)))

    # region: expert_act
    def act(self, state: DriveState) -> np.ndarray:
        """Devuelve (steer, throttle, brake)."""
        track, car, projection = state.track, state.car, state.projection
        speed = max(0.0, car.forward_speed)
        lateral = projection.lateral  # + = a la izquierda de la línea central
        nose_error = np.degrees(state.heading_error())  # + = la pista va a la izquierda de la trompa

        # Recuperación. Lento y contra un borde: una reversa corta. Solo lento: alinearse y acelerar.
        slow = speed < 1.5
        against_edge = abs(lateral) > 0.75 * track.half_width
        if self.memory:
            self.wall_jam_timer = self.wall_jam_timer + self.decision_dt if (slow and against_edge) else 0.0
            if self.wall_jam_timer > 1.0:
                self.escape_time_left, self.wall_jam_timer = 0.6, 0.0
            if self.escape_time_left > 0:
                self.escape_time_left -= self.decision_dt
                # En reversa el volante se invierte: girar al revés lleva la trompa hacia la pista.
                return np.array([clamp(-nose_error / 20, -1, 1), -1.0, 0.0])
        if slow:
            steer = clamp((nose_error - lateral * 2) / 18, -1, 1)
            self.steer_smooth = steer
            return np.array([steer, 1.0, 0.0])

        # Velocidad: acelerar hasta el objetivo, frenar si se pasa del margen, si no, ir de crucero.
        target = self.target_speed(state)
        if speed < target - 0.5:
            throttle, brake = 1.0, 0.0
        elif speed > target + brake_margin(state.aggression):
            throttle, brake = 0.0, clamp((speed - target) / 3, 0, 1)
        else:
            throttle, brake = lerp(0.45, 0.9, state.aggression), 0.0

        # Dirección: persecución pura hacia un punto por delante, más una corrección hacia el centro.
        heading_error = self.aim_error(state)
        centring = -clamp(lateral * 1.4, -13, 13)  # a la izquierda del centro → girar a la derecha
        if self.sign_bug:
            centring = -centring
        raw_steer = clamp((heading_error + centring) / 16, -1, 1)
        if self.memory:
            self.steer_smooth = lerp(self.steer_smooth, raw_steer, 0.35)
            steer = self.steer_smooth
        else:
            steer = raw_steer
        return np.array([steer, throttle, brake])
    # endregion: expert_act
