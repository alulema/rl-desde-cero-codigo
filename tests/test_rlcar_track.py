import numpy as np
import pytest

from rlcar.track import Track, fixed_circuit, random_circuit, resample_closed


def circle(radius: float, spacing: float = 1.0) -> Track:
    angles = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    points = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return Track(resample_closed(points, spacing), width=12.0)


def test_fixed_circuit_length_matches_formula():
    track = fixed_circuit(half_length=120, half_height=60, corner_radius=40)
    straights = 4 * (120 - 40) + 4 * (60 - 40)
    expected = straights + 2 * np.pi * 40
    assert track.length == pytest.approx(expected, rel=0.002)


def test_fixed_circuit_corner_radius():
    assert fixed_circuit(corner_radius=40).min_corner_radius() == pytest.approx(40, rel=0.02)


def test_curvature_of_a_circle():
    radii = circle(50).corner_radii()
    assert np.median(radii) == pytest.approx(50, rel=0.01)


def test_projection_on_start_straight():
    # La meta está en (0, −60) y el auto sale hacia +x: la izquierda es +y.
    track = fixed_circuit()
    p = track.project(np.array([10.0, -57.0]))
    assert p.progress == pytest.approx(10 / track.length, abs=1 / track.n)
    assert p.lateral == pytest.approx(3.0, abs=0.01)
    assert p.tangent == pytest.approx([1.0, 0.0], abs=1e-6)


def test_projection_right_of_centre_is_negative():
    track = fixed_circuit()
    assert track.project(np.array([10.0, -64.0])).lateral == pytest.approx(-4.0, abs=0.01)


def test_forward_metres_wraps_at_finish_line():
    track = fixed_circuit()
    assert track.forward_metres(0.995, 0.005) == pytest.approx(0.01 * track.length)
    assert track.forward_metres(0.005, 0.995) == pytest.approx(-0.01 * track.length)


def test_turn_is_positive_for_a_left_turn():
    # El circuito se recorre en sentido antihorario: todas las curvas son a la izquierda.
    track = fixed_circuit()
    corner_start = track.project(np.array([80.0, -60.0])).index
    assert track.turn_deg(corner_start, 0, 60) > 45


def test_random_circuit_is_reproducible_and_respects_min_radius():
    a = random_circuit(3, min_corner_radius=30)
    b = random_circuit(3, min_corner_radius=30)
    np.testing.assert_array_equal(a.points, b.points)
    assert a.min_corner_radius() >= 30


def test_technical_circuit_closes_with_expected_length():
    from rlcar.track import technical_circuit

    track = technical_circuit(straight=200, hairpin_radius=15)
    assert track.length == pytest.approx(2 * 200 + 2 * np.pi * 15, rel=0.002)
    assert track.min_corner_radius() == pytest.approx(15, rel=0.03)
