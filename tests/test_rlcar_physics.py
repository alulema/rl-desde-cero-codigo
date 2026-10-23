import numpy as np
import pytest

from rlcar.physics import CarParams, CarState, min_turn_radius, step, with_agentic_grip, with_grip_bug

DT = 0.02


def drive(state: CarState, action, seconds: float, params: CarParams) -> CarState:
    for _ in range(round(seconds / DT)):
        state = step(state, np.array(action, dtype=float), DT, params)
    return state


def speed(state: CarState) -> float:
    return float(np.hypot(state.vx, state.vy))


def test_full_throttle_from_rest():
    # 10 m/s² menos el arrastre: después de 1 s, algo menos de 10 m/s.
    state = drive(CarState(0, 0, 0, 0, 0), [0, 1, 0], 1.0, CarParams())
    assert 9.0 < state.forward_speed < 10.0
    assert state.y == pytest.approx(0.0)


def test_braking_distance():
    # De 20 m/s, a 15 m/s² (más un poco de arrastre): unos 13 m.
    state = CarState(0, 0, 0, 20, 0)
    while state.forward_speed > 0.2:
        state = step(state, np.array([0.0, 0.0, 1.0]), DT, CarParams())
    assert 11.0 < state.x < 14.0


def test_turn_radius_at_20_mps():
    # La cifra que usa la bitácora de Agentic Racing para las curvas imposibles.
    assert min_turn_radius(20.0, CarParams()) == pytest.approx(11.54, abs=0.01)


def test_steering_left_turns_left():
    state = drive(CarState(0, 0, 0, 15, 0), [0.3, 0.3, 0], 1.0, CarParams())
    assert state.heading > 0 and state.y > 0


def test_reverse_inverts_steering():
    state = drive(CarState(0, 0, 0, -5, 0), [0.5, 0, 0], 0.5, CarParams())
    assert state.heading < 0


def test_rotate_grip_does_not_create_speed():
    # Un giro brusco sin acelerar no puede terminar más rápido de lo que empezó.
    state = drive(CarState(0, 0, 0, 13, 0), [1, 0, 0], 1.0, CarParams(grip_model="rotate"))
    assert speed(state) < 13


def test_agentic_grip_creates_speed_when_turning():
    # El agarre de Agentic Racing tras el arreglo suma magnitudes: girar acelera el auto.
    state = drive(CarState(0, 0, 0, 13, 0), [1, 0, 0], 1.0, with_agentic_grip(CarParams()))
    assert speed(state) > 20


def test_grip_bug_brakes_the_car_when_turning():
    # El bug de antes del arreglo: borra la velocidad lateral y cada giro frena.
    state = drive(CarState(0, 0, 0, 13, 0), [1, 0, 0], 1.0, with_grip_bug(CarParams()))
    assert speed(state) < 7
