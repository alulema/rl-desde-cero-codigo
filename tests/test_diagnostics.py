"""Cada chequeo del capítulo 8 encuentra su bug plantado y no marca nada en el entorno sano."""

from dataclasses import replace

import numpy as np
import pytest

from algos.returns_gae import gae_batch
from rlcar import presets
from rlcar.diagnostics import (RandomPolicy, baseline_check, observation_check, reward_check, run_all,
                               steering_check, termination_check)
from rlcar.evaluate import ExpertPolicy, evaluate
from rlcar.rewards import end_terms

SEEDS = range(1000, 1003)
CLEAN = presets.technical()


@pytest.fixture
def expert():
    return ExpertPolicy(memory=False)


def test_zero_filled_observation_shows_as_constant_values(expert):
    # Con 3 episodios la agresión puede salir igual en todos: también sería constante. Con 5, no.
    seeds = range(1000, 1005)
    assert not observation_check(CLEAN, [RandomPolicy(), expert], seeds)["flag"]
    result = observation_check(presets.zero_filled_observation(CLEAN), [RandomPolicy(), expert], seeds)
    assert result["flag"]
    assert result["constant"][0] == "forward_speed" and len(result["constant"]) == 9
    assert set(result["constant_values"].values()) == {0.0}


def test_grip_bug_shows_as_speed_lost_when_turning():
    assert not steering_check(CLEAN.car)["flag"]
    result = steering_check(presets.grip_bug(CLEAN).car)
    assert result["flag"] and result["turn_speed_change"] < -0.2


def test_short_time_limit_is_below_the_baseline_speed(expert):
    assert not termination_check(CLEAN, expert, SEEDS)["flag"]
    result = termination_check(presets.short_time_limit(CLEAN), expert, SEEDS)
    assert result["flag"] and result["speed_needed"] > result["baseline_mean_speed"]


def test_sign_bug_baseline_does_not_return_to_the_centre(expert):
    assert not baseline_check(CLEAN, expert, SEEDS)["flag"]
    result = baseline_check(CLEAN, ExpertPolicy(memory=False, sign_bug=True), SEEDS)
    assert result["flag"] and result["remaining_offset"] > 1.0


def test_braking_does_not_pay_more_than_the_baseline_lap(expert):
    result = reward_check(CLEAN, expert, SEEDS)
    assert not result["flag"]
    assert result["returns"]["baseline"] > result["returns"]["brake"]


def test_chapter_8_exercises(expert):
    # Ejercicio 1: la velocidad media necesaria y el bono de una vuelta de 16 s con cada límite.
    lap = 0.99 * 494
    assert round(lap / 60, 1) == 8.2 and round(lap / 25, 1) == 19.6
    bonus = {limit: end_terms("lap", 1 - 16 / limit, CLEAN.reward)["lap"] for limit in (60, 25)}
    assert bonus[60] == pytest.approx(17.87, abs=0.01) and bonus[25] == pytest.approx(14.88, abs=0.01)
    # Ejercicio 2: el error de TD del último paso, si el episodio terminó y si lo cortó el reloj.
    rewards, values = np.ones((3, 1)), np.full((3, 1), 2.0)
    ended = np.array([[0.0], [0.0], [1.0]])
    for terminated, last in ((ended, -1.0), (np.zeros((3, 1)), 0.8)):
        advantages = gae_batch(rewards, values, values, terminated, ended, 0.9, 0.95)
        assert advantages[-1, 0] == pytest.approx(last)
    # Ejercicio 3: media y desviación (poblacional) de las vueltas con los cuatro bugs.
    laps = [19, 0, 19, 1, 0]
    assert np.mean(laps) == pytest.approx(7.8) and round(float(np.std(laps)), 1) == 9.2
    # Ejercicio 5: con el progreso en negativo, ningún chequeo marca nada; la suma por término sí lo ve.
    flipped = replace(CLEAN, reward=replace(CLEAN.reward, progress_per_metre=-0.02))
    seeds = range(1000, 1020)
    checks = run_all(flipped, expert, seeds)
    assert not any(check["flag"] for check in checks.values())
    returns = checks["reward"]["returns"]
    assert returns["brake"] == pytest.approx(-2.90, abs=0.005) and returns["idle"] == pytest.approx(-4.45, abs=0.005)
    records = evaluate(flipped, expert, seeds)
    assert np.mean([r.reward_terms["progress"] for r in records]) == pytest.approx(-9.79, abs=0.01)
    assert np.mean([r.episode_return for r in records]) == pytest.approx(8.35, abs=0.01)
    assert np.mean([r.reward_terms["lap"] for r in records]) == pytest.approx(16.03, abs=0.005)
    assert np.mean([r.reward_terms["target_speed"] for r in records]) == pytest.approx(4.03, abs=0.005)
