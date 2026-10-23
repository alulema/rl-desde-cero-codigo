import numpy as np
import pytest
import torch

from algos.ppo import PPOConfig, RunningNormalizer, clipped_objective, train
from rlcar import RLCarEnv, presets


@pytest.mark.parametrize(
    "ratio, advantage, expected",
    [
        (1.5, 1.0, 1.2),  # mejor de lo esperado y ya subió de más: el premio se corta en 1 + ε
        (0.5, 1.0, 0.5),  # mejor de lo esperado pero bajó: sin tope, conviene volver a subirla
        (0.5, -1.0, -0.8),  # peor de lo esperado y ya bajó de más: el premio de bajarla se corta en 1 − ε
        (1.5, -1.0, -1.5),  # peor de lo esperado pero subió: sin tope, el castigo pasa completo
    ],
)
def test_clipped_objective_four_cases(ratio, advantage, expected):
    value = clipped_objective(torch.tensor([ratio]), torch.tensor([advantage]), epsilon=0.2)
    assert value.item() == pytest.approx(expected)


def test_running_normalizer_matches_numpy():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=(100, 4)) for _ in range(5)]
    normalizer = RunningNormalizer(4)
    for batch in batches:
        normalizer.update(batch)
    everything = np.concatenate(batches)
    np.testing.assert_allclose(normalizer.mean, everything.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(normalizer.var, everything.var(axis=0), rtol=1e-3)


def make_env():
    return RLCarEnv(presets.agentic())


def test_training_runs_and_logs_finite_metrics():
    config = PPOConfig(total_steps=2 * 64 * 2, n_envs=2, n_steps=64, epochs=2, minibatch_size=32)
    agent, history = train(make_env, config, log=lambda row: None, vector="sync")
    assert len(history) == 2
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(history[-1][key])
    observation, _ = make_env().reset(seed=0)
    assert agent.act(observation).shape == (3,)


def test_chapter_2_ratio_from_log_probs_and_exercise():
    # La acción del ejercicio 4 del capítulo 0: log-densidad −0.616354 con la policy vieja y 0.258646 con
    # la nueva. La razón es 2.4; con A = +1, el objetivo se queda en 1.2.
    ratio = torch.tensor([-0.616354, 0.258646]).diff().exp()
    assert ratio.item() == pytest.approx(2.3989, abs=1e-4)
    assert clipped_objective(ratio, torch.tensor([1.0]), 0.2).item() == pytest.approx(1.2)
    # Ejercicio del capítulo 2, con A = ±2.
    cases = [(1.1, 2.0, 2.2), (0.7, 2.0, 1.4), (0.7, -2.0, -1.6), (1.3, -2.0, -2.6)]
    for r, a, expected in cases:
        value = clipped_objective(torch.tensor([r]), torch.tensor([a]), 0.2).item()
        assert value == pytest.approx(expected)
