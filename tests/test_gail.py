import numpy as np
import pytest

from algos.bc import Demonstrations
from algos.gail import GAIL, GAILConfig, gail_reward

LOW, HIGH = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0])


@pytest.mark.parametrize("d, expected", [(0.1, 0.105), (0.5, 0.693), (0.9, 2.303)])
def test_gail_reward_worked_values(d, expected):
    assert gail_reward(np.array([d]))[0] == pytest.approx(expected, abs=1e-3)


def discriminate(gradient_penalty: float):
    """Pares del experto y del agente que se separan por completo por la acción."""
    rng = np.random.default_rng(0)
    expert = Demonstrations(rng.normal(size=(1_000, 4)), np.tile([0.5, 1.0, 0.0], (1_000, 1)))
    gail = GAIL(expert, LOW, HIGH, GAILConfig(gradient_penalty=gradient_penalty))
    for _ in range(40):  # las métricas son de D antes de cada actualización
        rewards, stats = gail(rng.normal(size=(64, 8, 4)), np.tile([-0.5, -1.0, 1.0], (64, 8, 1)), np.zeros((64, 8)))
    return rewards, stats


def test_discriminator_separates_expert_from_agent():
    rewards, stats = discriminate(gradient_penalty=0.0)
    assert stats["d_expert"] > 0.95 and stats["d_agent"] < 0.05
    assert rewards.shape == (64, 8) and np.all(rewards > 0)  # siempre positiva: el sesgo de supervivencia


def test_gradient_penalty_keeps_the_discriminator_from_saturating():
    _, stats = discriminate(gradient_penalty=10.0)
    assert 0.6 < stats["d_expert"] < 0.9 and 0.1 < stats["d_agent"] < 0.4


def test_weights_mix_environment_and_imitation():
    rng = np.random.default_rng(1)
    expert = Demonstrations(rng.normal(size=(100, 4)), rng.uniform(0, 1, size=(100, 3)))
    only_env = GAIL(expert, LOW, HIGH, GAILConfig(gail_weight=0.0, env_weight=1.0))
    env_rewards = rng.normal(size=(16, 2))
    rewards, _ = only_env(rng.normal(size=(16, 2, 4)), rng.normal(size=(16, 2, 3)), env_rewards)
    np.testing.assert_allclose(rewards, env_rewards)


def test_chapter_4_reward_scale():
    """Con D = 0,5 y peso 0,15, GAIL paga 0,104 por decisión."""
    assert 0.15 * gail_reward(np.array([0.5]))[0] == pytest.approx(0.104, abs=1e-3)
    # En el último lote de GAIL 0,15 + entorno, la recompensa media de GAIL fue 0,44–0,54.
    assert 0.15 * 0.44 == pytest.approx(0.066) and 0.15 * 0.54 == pytest.approx(0.081)


def test_chapter_4_exercises():
    # 1. La recompensa a mano.
    for d, expected in [(0.2, 0.223), (0.7, 1.204), (0.99, 4.605)]:
        assert gail_reward(np.array([d]))[0] == pytest.approx(expected, abs=1e-3)
    # 2. La pérdida del discriminador para un par de cada lado (el texto del capítulo usa 0,8 y 0,3).
    assert -np.log(0.8) - np.log(1 - 0.3) == pytest.approx(0.580, abs=1e-3)
    loss = -np.log(0.6) - np.log(1 - 0.45)
    assert loss == pytest.approx(1.109, abs=1e-3)
    # 3. La escala: D = 0,35 con peso 0,15, por decisión y en una vuelta de 300 decisiones.
    per_decision = 0.15 * gail_reward(np.array([0.35]))[0]
    assert per_decision == pytest.approx(0.0646, abs=1e-4)
    assert 300 * per_decision == pytest.approx(19.4, abs=0.05)
    # 4. Sobrevivir: GAIL solo, D = 0,25. Una vuelta de 270 decisiones contra 600 decisiones lentas.
    r = gail_reward(np.array([0.25]))[0]
    assert 270 * r == pytest.approx(77.67, abs=0.01)
    assert 600 * r == pytest.approx(172.6, abs=0.1)
    d_break_even = 1 - np.exp(-270 * r / 600)
    assert d_break_even == pytest.approx(0.1214, abs=1e-4)
