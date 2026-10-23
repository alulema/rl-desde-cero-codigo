import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from algos.bc import Demonstrations
from algos.combine import BCAuxConfig, BCAuxLoss
from algos.ppo import ActorCritic, PPOConfig, train
from rlcar import RLCarEnv, presets

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_bc_aux_pulls_the_actor_towards_the_expert_and_switches_off():
    rng = np.random.default_rng(0)
    demos = Demonstrations(rng.normal(size=(512, 4)), np.tile([0.3, 0.8, 0.0], (512, 1)))
    model = ActorCritic(4, 3, PPOConfig())
    hook = BCAuxLoss(demos, ppo_learning_rate=3e-3, config=BCAuxConfig(active_fraction=0.5))
    first = hook(model, lambda x: x, progress=0.0)["bc_loss"]
    for _ in range(10):
        last = hook(model, lambda x: x, progress=0.1)["bc_loss"]
    assert last < first / 4
    before = [p.clone() for p in model.actor.parameters()]
    stats = hook(model, lambda x: x, progress=0.5)
    assert math.isnan(stats["bc_loss"]) and stats["bc_learning_rate"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))


def make_env():
    return RLCarEnv(presets.technical())


def test_train_can_start_from_a_trained_agent():
    config = PPOConfig(total_steps=512, n_envs=2, n_steps=128, epochs=1)
    first, _ = train(make_env, config, log=lambda row: None, vector="sync")
    second, _ = train(make_env, PPOConfig(total_steps=256, n_envs=2, n_steps=128, epochs=1, learning_rate=0.0),
                      log=lambda row: None, vector="sync", init=first)
    # Con tasa 0 la red no cambia; el normalizador sí, porque sigue sumando observaciones.
    for a, b in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(a, b)
    assert second.normalizer.count > first.normalizer.count
    assert second.normalizer is not first.normalizer


def test_chapter_5_exercises():
    # Ejercicio 1: la tasa de BC, desde el propio hook.
    rng = np.random.default_rng(0)
    demos = Demonstrations(rng.normal(size=(64, 4)), np.zeros((64, 3)))
    model = ActorCritic(4, 3, PPOConfig())
    hook = BCAuxLoss(demos, ppo_learning_rate=3e-4, config=BCAuxConfig(minibatch_size=8))
    rates = [hook(model, lambda x: x, progress=p)["bc_learning_rate"] for p in (0.0, 0.1, 0.3)]
    assert rates == pytest.approx([1.5e-4, 7.5e-5, 0.0])
    assert BCAuxConfig().active_fraction * 500_000 == 100_000

    # Ejercicio 2: la tasa de PPO en el último lote de D y en el primero de E.
    n_updates = 500_000 // (PPOConfig().n_envs * PPOConfig().n_steps)
    assert n_updates == 244
    last_d = 3e-4 * (1 - 243 / n_updates)
    assert last_d == pytest.approx(1.23e-6, abs=5e-9)
    assert 3e-4 / last_d == pytest.approx(244)

    # Ejercicio 3: medias, rangos y pares de "conservar − atacar".
    c = [6.6, 5.0, 6.1, 5.4, 0.6]
    d = [5.9, 5.2, 3.3, 2.5, 3.8]
    e = [0.5, -0.1, 1.2, 0.0, 1.8]
    assert [round(float(np.mean(x)), 1) for x in (c, d, e)] == [4.7, 4.1, 0.7]
    assert min(c) < max(e)  # C y E se solapan
    assert min(d) > max(e)  # D y E no
    results = json.loads((RESULTS / "ch05_combine.json").read_text())
    drops = [run["variants"]["D"]["conserve_minus_attack_s"]
             - run["variants"]["E"]["conserve_minus_attack_s"] for run in results["runs"]]
    assert round(min(drops), 1) == 1.9 and round(max(drops), 1) == 5.4
