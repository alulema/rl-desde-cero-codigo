from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlcar import RLCarEnv, presets
from rlcar.env import EnvConfig, SpawnConfig
from rlcar.evaluate import ExpertPolicy, run_episode
from rlcar.observations import ObsConfig, build_observation, observation_names
from rlcar.physics import CarState
from rlcar.rewards import RewardConfig, end_terms, step_terms
from rlcar.sensors import cast_rays
from rlcar.state import DriveState
from rlcar.terminations import TerminationConfig, TerminationTracker
from rlcar.track import fixed_circuit

TRACK = fixed_circuit()


def state_at(x: float, y: float, heading: float = 0.0, speed: float = 0.0) -> DriveState:
    car = CarState(x, y, heading, speed * np.cos(heading), speed * np.sin(heading))
    return DriveState(track=TRACK, car=car, projection=TRACK.project(car.position), aggression=0.5)


# --- Sensores y observación ------------------------------------------------------------------------------

def test_ray_straight_ahead_and_sideways_on_start_straight():
    # En la recta de salida, centrado: el rayo de la izquierda (90°) toca el borde a 6 m.
    rays = cast_rays(np.array([0.0, -60.0]), 0.0, TRACK, TRACK.project(np.array([0.0, -60.0])).index,
                     n_rays=3, fov_deg=180, length=70)
    assert rays[2] == pytest.approx(6 / 70, abs=1e-6)  # izquierda
    assert rays[0] == pytest.approx(6 / 70, abs=1e-6)  # derecha
    assert rays[1] == 1.0  # adelante, la recta sigue más de 70 m


def test_observation_size_follows_config():
    env = RLCarEnv(EnvConfig(obs=ObsConfig(progress=False, previous_steer=True, ray_format="mlagents")))
    observation, _ = env.reset(seed=0)
    assert observation.shape == (9 * 3 + 2 + 1 + 1 + 3 + 1 + 1,)
    assert len(env.observation_names) == observation.shape[0]
    assert "progress" not in env.observation_names


def test_check_env_passes():
    check_env(RLCarEnv(presets.agentic()), skip_render_check=True)


def test_same_seed_same_trajectory():
    def trajectory(seed):
        env = RLCarEnv(presets.agentic())
        env.reset(seed=seed)
        rng = np.random.default_rng(123)
        return [env.step(rng.uniform([-1, -1, 0], [1, 1, 1]))[0] for _ in range(30)]

    np.testing.assert_array_equal(trajectory(7), trajectory(7))


# --- Recompensa -------------------------------------------------------------------------------------------

def test_reward_terms_sum_to_reward_and_zero_weights_are_absent():
    env = RLCarEnv(presets.agentic())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.0, 1.0, 0.0]))
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert "time" not in info["reward_terms"] and "alive" not in info["reward_terms"]


def test_progress_term_pays_per_metre():
    terms = step_terms(state_at(0, -60), 2.0, False, 0.02, RewardConfig(progress_per_metre=0.02), 55)
    assert terms == {"progress": pytest.approx(0.04)}


def test_target_speed_term_peaks_at_target():
    config = RewardConfig(target_speed_per_sec=0.25)
    # En la mitad de la recta de 160 m, sin curva en los próximos 55 m: objetivo = 0.42 · 55 = 23.1 m/s.
    at_target = step_terms(state_at(-50, -60, speed=23.1), 0.0, False, 1.0, config, 55)["target_speed"]
    too_slow = step_terms(state_at(-50, -60, speed=10.0), 0.0, False, 1.0, config, 55)["target_speed"]
    assert at_target == pytest.approx(0.25, abs=1e-3)
    assert too_slow < at_target


def test_edge_term_ramps_from_safe_fraction():
    config = RewardConfig(edge_per_sec=0.6)
    assert "edge" not in step_terms(state_at(0, -60 + 3.0), 0, False, 1.0, config, 55)  # 0.5 del medio ancho
    assert step_terms(state_at(0, -60 + 6.0), 0, False, 1.0, config, 55)["edge"] == pytest.approx(-0.6)


def test_lap_bonus_scales_with_time_left():
    config = RewardConfig(lap_bonus=12, fast_lap_bonus=8)
    assert end_terms("lap", 0.5, config) == {"lap": pytest.approx(16.0)}
    assert end_terms("off_track", 0.5, RewardConfig(off_track=1.0)) == {"end_off_track": -1.0}


# --- Terminaciones ----------------------------------------------------------------------------------------

def test_off_track_beyond_margin():
    tracker = TerminationTracker(TerminationConfig())
    assert tracker.update(state_at(0, -60 + 7.9, speed=10), 0.0, 0.02) is None
    assert tracker.update(state_at(0, -60 + 8.1, speed=10), 0.0, 0.02) == "off_track"


def test_stuck_only_after_moving_and_after_the_timeout():
    tracker = TerminationTracker(TerminationConfig(stuck_seconds=3.0, stall_seconds=None))
    stopped = state_at(0, -60, speed=0.0)
    for _ in range(200):  # 4 s detenido sin haber arrancado: no cuenta
        assert tracker.update(stopped, 0.0, 0.02) is None
    tracker.update(state_at(0, -60, speed=5.0), 0.0, 0.02)
    reasons = [tracker.update(stopped, 0.0, 0.02) for _ in range(160)]
    assert reasons[148] is None and "stuck" in reasons


def test_stall_ends_episode_without_progress():
    tracker = TerminationTracker(TerminationConfig(stall_seconds=5.0, stall_min_metres=8.0))
    state = state_at(0, -60, speed=1.0)
    reasons = []
    for _ in range(int(5.5 / 0.02)):
        state.time += 0.02
        reasons.append(tracker.update(state, 0.01, 0.02))  # 0.5 m/s: 2.5 m en 5 s
    assert "stall" in reasons


def test_time_limit_truncates_instead_of_terminating():
    config = replace(presets.agentic(), termination=TerminationConfig(time_limit_seconds=1.0, stall_seconds=None))
    env = RLCarEnv(config)
    env.reset(seed=0)
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.2, 0.0]))
    assert truncated and not terminated and info["end_reason"] == "time_limit"


# --- Experto ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("memory", [True, False])
def test_expert_completes_a_lap_on_the_fixed_circuit(memory):
    config = replace(presets.agentic(), aggression=0.5, spawn=SpawnConfig(clean=True))
    record, _ = run_episode(config, ExpertPolicy(memory=memory), seed=0)
    assert record.end_reason == "lap"


def test_expert_is_faster_when_more_aggressive():
    times = []
    for aggression in (0.15, 0.85):
        config = replace(presets.agentic(), aggression=aggression, spawn=SpawnConfig(clean=True))
        times.append(run_episode(config, ExpertPolicy(), seed=0)[0].lap_time)
    assert times[1] < times[0]


def test_race04_stopping_ends_the_episode_with_a_positive_return():
    # El atajo de la cuarta corrida: avanzar, detenerse y aceptar el −1. El return sigue siendo positivo.
    config = replace(presets.race04(), aggression=0.5)
    env = RLCarEnv(config)
    env.reset(seed=0)
    expert = ExpertPolicy().expert
    total = 0.0
    while True:
        action = expert.act(env.state) if env.state.lap_metres < 150 else np.array([0.0, 0.0, 1.0])
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    assert info["end_reason"] == "stuck"
    assert info["reward_terms"]["end_stuck"] == -1.0
    assert total > 0


def test_expert_laps_the_technical_circuit():
    config = replace(presets.technical(), aggression=0.5)
    assert run_episode(config, ExpertPolicy(), seed=0)[0].end_reason == "lap"


def test_stacked_observations_keep_the_previous_ones():
    config = presets.technical()
    config = replace(config, obs=replace(config.obs, stack=3))
    env = RLCarEnv(config)
    first, _ = env.reset(seed=0)
    n = len(first) // 3
    assert env.observation_space.shape == (3 * n,) and len(env.observation_names) == 3 * n
    np.testing.assert_array_equal(first[:n], first[n:2 * n])  # al arrancar se repite la primera
    second, *_ = env.step(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(second[n:2 * n], first[:n])
    third, *_ = env.step(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(third[2 * n:], first[:n])


def per_second_reward(speed: float, target: float, target_weight: float = 0.25) -> float:
    """La recompensa por segundo de `agentic()` en la recta de salida, centrado y alineado."""
    config = replace(presets.agentic().reward, straight_speed_frac=target / 55,
                     target_speed_per_sec=target_weight)
    return sum(step_terms(state_at(0, -60, speed=speed), speed, False, 1.0, config, 55).values())


def test_chapter_6_speed_threshold():
    # La horquilla (6.6 m/s): un máximo local de 0.39/s, que el progreso solo supera desde 18.6 m/s.
    assert per_second_reward(6.6, 6.6) == pytest.approx(0.388, abs=1e-3)
    assert per_second_reward(7.1, 6.6) < per_second_reward(6.6, 6.6)
    assert per_second_reward(18.55, 6.6) < per_second_reward(6.6, 6.6) < per_second_reward(18.65, 6.6)
    # La recta (23.1 m/s): por encima del umbral, pasarse del objetivo paga más.
    assert per_second_reward(23.6, 23.1) > per_second_reward(23.1, 23.1)
    assert 0.25 * 1.3 / 0.02 == pytest.approx(16.25)
    assert 0.25 * 1.3 / (0.02 + 0.05 / 55) == pytest.approx(15.55, abs=0.01)


def test_chapter_6_exercises():
    # Ejercicio 1: el bono de vuelta para 29.5 s y 16.5 s, con un límite de 60 s.
    reward = presets.agentic().reward
    assert end_terms("lap", 30.5 / 60, reward)["lap"] == pytest.approx(16.07, abs=0.005)
    assert end_terms("lap", 43.5 / 60, reward)["lap"] == pytest.approx(17.80)
    assert 0.02 * 0.99 * 494.22 == pytest.approx(9.79, abs=0.005)
    # Ejercicio 2: con peso 0.5, el umbral sube y la recta tiene un máximo local hasta los 47 m/s.
    assert 0.5 * 1.3 / 0.02 == pytest.approx(32.5)
    assert round(0.5 * 1.3 / (0.02 + 0.05 / 55), 1) == 31.1
    assert per_second_reward(23.1, 23.1, 0.5) == pytest.approx(0.983, abs=1e-3)
    assert per_second_reward(24.0, 23.1, 0.5) < per_second_reward(23.1, 23.1, 0.5)
    assert per_second_reward(46.9, 23.1, 0.5) < per_second_reward(23.1, 23.1, 0.5)
    assert per_second_reward(47.1, 23.1, 0.5) > per_second_reward(23.1, 23.1, 0.5)
    # Ejercicio 3: el atajo con race04 y la duración del episodio.
    assert 5.11 + 0.66 + 0.36 - 0.03 - 1 - 0.47 == pytest.approx(4.63)
    assert 0.47 / (0.0005 * 50) == pytest.approx(18.8)
    # Ejercicio 4: seguir vivo.
    assert 0.5 * 29.5 == pytest.approx(14.75) and 30 - 14.75 == pytest.approx(15.25)


def test_chapter_7_exercises():
    # Ejercicio 1: la observación de `rlcar`, con el auto 4 m a la izquierda del centro.
    car = CarState(0.0, -60.0 + 4.0, 0.0, 30.0, 2.0)
    state = DriveState(track=TRACK, car=car, projection=TRACK.project(car.position), aggression=0.5)
    values = dict(zip(observation_names(ObsConfig()), build_observation(state, ObsConfig(), 55.0)))
    assert values["forward_speed"] == pytest.approx(0.545, abs=1e-3)
    assert values["lateral_speed"] == pytest.approx(0.036, abs=1e-3)
    assert values["lateral_offset"] == pytest.approx(0.667, abs=1e-3)
    assert 18 / 180 == pytest.approx(0.1) and 45 / 90 == pytest.approx(0.5)
    far = CarState(0.0, -60.0 + 14.0, 0.0, 30.0, 2.0)
    far_state = DriveState(track=TRACK, car=far, projection=TRACK.project(far.position), aggression=0.5)
    assert build_observation(far_state, ObsConfig(), 55.0)[12] == pytest.approx(2.0)
    # Ejercicio 2: la meta del circuito técnico.
    assert 493 / 494 == pytest.approx(0.998, abs=5e-4) and 1 / 494 == pytest.approx(0.002, abs=5e-4)
    # Ejercicio 3: medias y desviaciones (poblacionales) de las vueltas por semilla.
    full, no_progress, previous = [8, 9, 16, 4, 18], [16, 8, 14, 19, 7], [18, 19, 17, 17, 17]
    assert [round(float(np.mean(x)), 1) for x in (full, no_progress, previous)] == [11.0, 12.8, 17.6]
    assert [round(float(np.std(x)), 1) for x in (full, no_progress, previous)] == [5.2, 4.6, 0.8]


class _RandomNoBrake:
    """El agente al azar del capítulo 1 sin freno, como en `experiments/exercises.py`."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, observation, env):
        action = self.rng.uniform(env.action_space.low, env.action_space.high)
        action[2] = 0.0
        return action


def test_chapter_1_code_exercise():
    # Sin freno, el agente al azar ya no se detiene enseguida: se sale de la pista o deja de avanzar.
    records = [run_episode(presets.agentic(), _RandomNoBrake(seed), seed)[0] for seed in range(5)]
    assert not any(r.completed_lap for r in records)
    assert sorted(r.end_reason for r in records) == ["off_track", "off_track", "off_track", "stall", "stall"]
    assert all(r.mean_brake == 0.0 for r in records)


def test_progress_sincos_has_no_jump_at_the_finish():
    config = ObsConfig(progress_encoding="sincos")
    names = observation_names(config)
    assert "progress" not in names and {"progress_sin", "progress_cos"} <= set(names)
    env = RLCarEnv(EnvConfig(obs=config))
    observation, _ = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape


def test_discounted_return_matches_the_plain_sum_without_discount():
    plain, _ = run_episode(presets.technical(), ExpertPolicy(memory=False), 1000, gamma=1.0)
    discounted, _ = run_episode(presets.technical(), ExpertPolicy(memory=False), 1000)
    assert plain.discounted_return == pytest.approx(plain.episode_return)
    assert 0 < discounted.discounted_return < discounted.episode_return
