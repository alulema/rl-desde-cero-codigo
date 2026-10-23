import numpy as np
import pytest

from algos.returns_gae import discounted_return, gae, gae_batch, returns_to_go, td_errors

# El ejemplo resuelto del libro: tres pasos de avance (+1) y una salida de pista (−1), fin del episodio.
REWARDS = np.array([1.0, 1.0, 1.0, -1.0])
VALUES = np.array([2.0, 1.0, 0.5, -0.5])
GAMMA, LAM = 0.9, 0.8


def test_discounted_return_small_episode():
    # 1 + 0.5·1 + 0.25·1 = 1.75
    assert discounted_return(np.array([1.0, 1.0, 1.0]), gamma=0.5) == pytest.approx(1.75)


def test_weight_of_distant_reward_in_agentic_racing():
    # Agentic Racing decide 10 veces por segundo con γ = 0.995.
    # Una recompensa a 20 s (200 decisiones) pesa ≈ 0.37; a 90 s (900 decisiones), ≈ 0.01.
    at_20s = np.zeros(201)
    at_20s[200] = 1.0
    at_90s = np.zeros(901)
    at_90s[900] = 1.0
    assert discounted_return(at_20s, gamma=0.995) == pytest.approx(0.37, abs=0.005)
    assert discounted_return(at_90s, gamma=0.995) == pytest.approx(0.01, abs=0.005)


def test_worked_example_returns():
    # G_3 = −1;  G_2 = 1 + 0.9·(−1) = 0.1;  G_1 = 1 + 0.9·0.1 = 1.09;  G_0 = 1 + 0.9·1.09 = 1.981
    np.testing.assert_allclose(returns_to_go(REWARDS, GAMMA), [1.981, 1.09, 0.1, -1.0])
    assert returns_to_go(REWARDS, GAMMA)[0] == pytest.approx(discounted_return(REWARDS, GAMMA))


def test_worked_example_td_errors():
    # δ_0 = 1 + 0.9·1.0 − 2.0 = −0.1      (Q ≈ 1.9 frente a V = 2.0: el paso salió peor de lo esperado)
    # δ_1 = 1 + 0.9·0.5 − 1.0 = 0.45
    # δ_2 = 1 + 0.9·(−0.5) − 0.5 = 0.05
    # δ_3 = −1 + 0 − (−0.5) = −0.5        (estado final: no hay V(s_4))
    np.testing.assert_allclose(td_errors(REWARDS, VALUES, 0.0, GAMMA), [-0.1, 0.45, 0.05, -0.5])


def test_worked_example_gae():
    # γλ = 0.72.  A_3 = −0.5;  A_2 = 0.05 + 0.72·(−0.5) = −0.31;
    # A_1 = 0.45 + 0.72·(−0.31) = 0.2268;  A_0 = −0.1 + 0.72·0.2268 = 0.063296
    np.testing.assert_allclose(gae(REWARDS, VALUES, 0.0, GAMMA, LAM), [0.063296, 0.2268, -0.31, -0.5])


def test_gae_extremes():
    # λ = 0: la ventaja es el error TD.  λ = 1: la ventaja es G_t − V(s_t).
    np.testing.assert_allclose(gae(REWARDS, VALUES, 0.0, GAMMA, 0.0), td_errors(REWARDS, VALUES, 0.0, GAMMA))
    np.testing.assert_allclose(gae(REWARDS, VALUES, 0.0, GAMMA, 1.0), returns_to_go(REWARDS, GAMMA) - VALUES)


def test_truncation_bootstraps_with_the_value_of_the_last_state():
    # Si el episodio se cortó por tiempo, el futuro existía: G_3 = −1 + 0.9·V(s_4).
    assert returns_to_go(REWARDS, GAMMA, bootstrap=3.0)[-1] == pytest.approx(-1 + 0.9 * 3.0)


def test_gae_batch_matches_single_episode_and_does_not_cross_episodes():
    # Dos episodios seguidos en un mismo entorno: el ejemplo (termina) y luego un tramo que se corta por
    # tiempo con V(s_final) = 3.0.
    rewards = np.concatenate([REWARDS, [0.5, 0.5]])[:, None]
    values = np.concatenate([VALUES, [1.0, 1.2]])[:, None]
    next_values = np.array([1.0, 0.5, -0.5, 99.0, 1.2, 3.0])[:, None]  # 99: el siguiente episodio, no se usa
    terminated = np.array([0, 0, 0, 1, 0, 0], dtype=float)[:, None]
    ended = np.array([0, 0, 0, 1, 0, 1], dtype=float)[:, None]
    advantages = gae_batch(rewards, values, next_values, terminated, ended, GAMMA, LAM)
    np.testing.assert_allclose(advantages[:4, 0], gae(REWARDS, VALUES, 0.0, GAMMA, LAM))
    np.testing.assert_allclose(advantages[4:, 0], gae(np.array([0.5, 0.5]), np.array([1.0, 1.2]), 3.0, GAMMA, LAM))


def test_worked_example_q_and_advantage():
    # Capítulo 1. Q(s_t, a_t) estimado con un paso: r_t + γ V(s_{t+1}) (0 después del último paso).
    q = REWARDS + GAMMA * np.append(VALUES[1:], 0.0)
    np.testing.assert_allclose(q, [1.9, 1.45, 0.55, -1.0])
    # La ventaja A = Q − V con esa estimación es el error TD del capítulo 2.
    np.testing.assert_allclose(q - VALUES, td_errors(REWARDS, VALUES, 0.0, GAMMA))
    # Con el return real del episodio en lugar de Q: A = G − V. Las dos estimaciones no coinciden.
    np.testing.assert_allclose(returns_to_go(REWARDS, GAMMA) - VALUES, [-0.019, 0.09, -0.4, -0.5])


def test_chapter_1_exercises():
    # Ejercicio 1: el mismo episodio con γ = 0.99.
    assert discounted_return(REWARDS, 0.99) == pytest.approx(1.999801)
    # Ejercicio 2: una recompensa a 60 s (600 decisiones) con γ = 0.995.
    assert 0.995**600 == pytest.approx(0.0494, abs=1e-4)
    # Ejercicio 2: con γ = 0.9 por decisión, a los 2 s (20 decisiones) ya pesa 0.12.
    assert 0.9**20 == pytest.approx(0.1216, abs=1e-4)


def test_chapter_1_exercise_better_critic():
    # Ejercicio 3: si el crítico estima V(s_3) = −1, Q − V del paso 2 pasa a −0.4, igual que G − V.
    values = np.array([2.0, 1.0, 0.5, -1.0])
    q = REWARDS + GAMMA * np.append(values[1:], 0.0)
    np.testing.assert_allclose((q - values)[2:], [-0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose((returns_to_go(REWARDS, GAMMA) - values)[2:], [-0.4, 0.0], atol=1e-12)


def test_chapter_2_gae_half_life_and_exercises():
    # Con γ = 0.995 y λ = 0.95, el peso de un δ se reduce a la mitad cada ≈ 12 decisiones.
    assert np.log(0.5) / np.log(0.995 * 0.95) == pytest.approx(12.32, abs=0.01)
    assert np.log(0.5) / np.log(0.995 * 0.9) == pytest.approx(6.28, abs=0.01)
    assert np.log(0.5) / np.log(0.995 * 0.99) == pytest.approx(46.0, abs=0.05)
    # Pasos de gradiente: 244 lotes × 80 en rlcar; 488 × 30 en Agentic Racing.
    assert (500_000 // 2048) * 8 * 10 == 19_520 and (10_000_000 // 20_480) * 10 * 3 == 14_640
    # Ejercicio: GAE del episodio de 4 pasos con λ = 0.5 (γλ = 0.45).
    np.testing.assert_allclose(gae(REWARDS, VALUES, 0.0, GAMMA, 0.5), [0.0670625, 0.37125, -0.175, -0.5])
