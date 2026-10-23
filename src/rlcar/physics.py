"""Física del auto: una adaptación a 2D de `CarController` de Agentic Racing.

Los parámetros por defecto son los de `VehicleConfig`, para que las cuentas del libro valgan en los dos
lados, con una excepción: el agarre. El de Agentic Racing crea velocidad al girar, y aquí el modelo por
defecto la conserva (ver `grip_model`). Convención: ángulos en radianes, sentido antihorario positivo,
`steer > 0` = izquierda.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CarParams:
    mass: float = 1200.0  # kg
    linear_drag: float = 0.2  # como el linearDamping de Unity
    engine_force: float = 12000.0  # N  → 10 m/s² a fondo
    brake_force: float = 18000.0  # N  → 15 m/s²
    coast_force: float = 2500.0  # N  → ≈ 2.1 m/s² sin acelerador
    max_speed: float = 55.0  # m/s hacia adelante (~200 km/h)
    max_reverse_speed: float = 12.0  # m/s
    turn_rate_deg: float = 130.0  # °/s con el volante a fondo y autoridad completa
    high_speed_turn_factor: float = 0.35  # autoridad que queda a max_speed
    steer_fade_in_speed: float = 0.4  # m/s a los que la autoridad llega a 1
    min_steer_authority: float = 0.25  # autoridad con el auto detenido
    lateral_grip: float = 7.0  # fracción de velocidad lateral que se quita por segundo
    # region: grip_model_param
    # Qué hace el agarre con la velocidad lateral que quita:
    #   "redirect": devuelve grip_redirect de esa magnitud hacia adelante (Agentic Racing tras el arreglo;
    #               suma magnitudes en vez de rotar el vector, así que girar crea velocidad).
    #   "erase":    la borra (Agentic Racing antes del arreglo: cada giro frena el auto).
    #   "rotate":   gira la velocidad hacia la trompa sin cambiar su módulo.
    grip_model: str = "rotate"
    grip_redirect: float = 0.85
    # endregion: grip_model_param


@dataclass(frozen=True)
class CarState:
    x: float
    y: float
    heading: float  # radianes
    vx: float  # velocidad en el marco del mundo, m/s
    vy: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def forward(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def left(self) -> np.ndarray:
        return np.array([-np.sin(self.heading), np.cos(self.heading)])

    @property
    def forward_speed(self) -> float:
        """Velocidad en el sentido de la trompa, con signo (negativa = en reversa)."""
        return self.vx * np.cos(self.heading) + self.vy * np.sin(self.heading)

    @property
    def lateral_speed(self) -> float:
        """Velocidad hacia la izquierda del auto (derrape)."""
        return -self.vx * np.sin(self.heading) + self.vy * np.cos(self.heading)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# region: steer_authority
def steer_authority(speed: float, params: CarParams) -> float:
    """Qué fracción del giro máximo está disponible a esta velocidad (ApplySteering).

    Algo de giro aun detenido, autoridad completa desde `steer_fade_in_speed`, y menos a alta velocidad.
    """
    fade_in = lerp(params.min_steer_authority, 1.0, clamp(speed / max(0.01, params.steer_fade_in_speed), 0, 1))
    high_speed = clamp(speed / params.max_speed, 0, 1)
    return fade_in * lerp(1.0, params.high_speed_turn_factor, high_speed)
# endregion: steer_authority


def min_turn_radius(speed: float, params: CarParams) -> float:
    """Radio de giro con el volante a fondo, si el auto siguiera exactamente su trompa."""
    yaw_rate = np.radians(params.turn_rate_deg) * steer_authority(speed, params)
    return speed / yaw_rate


def drive_acceleration(forward_speed: float, throttle: float, brake: float, params: CarParams) -> float:
    """Aceleración longitudinal en m/s² (ApplyDrive). El freno manda sobre el acelerador."""
    if brake > 0.01 and abs(forward_speed) > 0.1:
        return -np.sign(forward_speed) * params.brake_force * clamp(brake, 0, 1) / params.mass
    if abs(throttle) > 0.01:
        over_limit = (throttle > 0 and forward_speed > params.max_speed) or (
            throttle < 0 and -forward_speed > params.max_reverse_speed
        )
        return 0.0 if over_limit else params.engine_force * throttle / params.mass
    if abs(forward_speed) > 0.1:
        return -np.sign(forward_speed) * params.coast_force / params.mass
    return 0.0


# region: physics_step
def step(state: CarState, action: np.ndarray, dt: float, params: CarParams) -> CarState:
    """Avanza un paso de física. `action` = (steer, throttle, brake), ya recortada a su rango.

    El orden es el de `CarController.FixedUpdate` + el paso de PhysX: la tracción, el giro y el agarre se
    calculan con el rumbo de antes del paso; después se integran la fuerza, el arrastre y la posición.
    """
    steer, throttle, brake = action
    forward, left = state.forward, state.left
    velocity = np.array([state.vx, state.vy])
    forward_speed = state.forward_speed

    acceleration = drive_acceleration(forward_speed, throttle, brake, params)

    yaw = steer * np.radians(params.turn_rate_deg) * steer_authority(abs(forward_speed), params) * dt
    if forward_speed < -0.1:
        yaw = -yaw  # en reversa, el volante se invierte, como en un auto real

    # Agarre lateral: quita una fracción de la velocidad de costado (ApplyLateralGrip).
    k = clamp(params.lateral_grip * dt, 0, 1)
    side_change = -state.lateral_speed * k
    speed_before = np.hypot(state.vx, state.vy)
    velocity = velocity + left * side_change
    if params.grip_model == "redirect":
        velocity = velocity + forward * abs(side_change) * params.grip_redirect
    elif params.grip_model == "rotate" and speed_before > 1e-9:
        velocity = velocity / np.linalg.norm(velocity) * speed_before

    velocity = velocity + forward * acceleration * dt
    velocity = velocity / (1 + params.linear_drag * dt)
    position = state.position + velocity * dt
    return CarState(position[0], position[1], state.heading + yaw, velocity[0], velocity[1])
# endregion: physics_step


def with_agentic_grip(params: CarParams) -> CarParams:
    """El agarre de Agentic Racing tras el arreglo: redirige, pero girar le suma velocidad al auto."""
    return replace(params, grip_model="redirect", grip_redirect=0.85, lateral_grip=7.0)


def with_grip_bug(params: CarParams) -> CarParams:
    """El agarre de Agentic Racing antes del arreglo: borra la velocidad lateral en lugar de redirigirla."""
    return replace(params, grip_model="erase", lateral_grip=9.0)
