"""Los ejemplos resueltos a mano del capítulo 0."""

import numpy as np
import pytest
import torch

from algos.gail import gail_reward
from algos.ml_minimo import (gradient_descent, mse, mse_gradient, normal_density, normal_log_density, predict,
                             sigmoid, standardize, unstandardize_weights)

# Dos momentos del experto: apunta 10° a la izquierda y gira 0.625; apunta 8° a la derecha y gira −0.5.
X = np.array([[10.0], [-8.0]])
Y = np.array([0.625, -0.5])


def test_loss_and_gradient_by_hand():
    w, b = np.array([0.0]), 0.0
    assert mse(predict(w, b, X), Y) == pytest.approx(0.3203125)
    grad_w, _ = mse_gradient(w, b, X, Y)
    assert grad_w[0] == pytest.approx(-10.25)


def test_one_step_with_a_good_and_a_too_large_learning_rate():
    for learning_rate, expected_w, expected_loss in ((0.005, 0.05125, 0.010378125), (0.02, 0.205, 1.6651125)):
        w = np.array([0.0]) - learning_rate * mse_gradient(np.array([0.0]), 0.0, X, Y)[0]
        assert w[0] == pytest.approx(expected_w)
        assert mse(predict(w, 0.0, X), Y) == pytest.approx(expected_loss)


def test_the_best_weight_is_the_one_the_expert_uses():
    # Sin sesgo, el mejor peso es Σxy / Σx² = 10.25 / 164 = 1/16.
    assert (X[:, 0] @ Y) / (X[:, 0] @ X[:, 0]) == pytest.approx(1 / 16)


def test_gradient_matches_autograd():
    # La regla de la cadena a mano, contra la que calcula PyTorch.
    w = torch.tensor([0.03], requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    loss = ((torch.as_tensor(X, dtype=torch.float32) @ w + b - torch.as_tensor(Y, dtype=torch.float32)) ** 2).mean()
    loss.backward()
    grad_w, grad_b = mse_gradient(np.array([0.03]), 0.1, X, Y)
    assert w.grad.item() == pytest.approx(grad_w[0], rel=1e-5)
    assert b.grad.item() == pytest.approx(grad_b, rel=1e-5)


def test_gradient_descent_recovers_a_known_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2)) * [20.0, 3.0]
    y = x @ np.array([1 / 16, -1.4 / 16]) + 0.1
    xs, mean, std = standardize(x)
    w, b, history = gradient_descent(xs, y, learning_rate=0.1, steps=300)
    w, b = unstandardize_weights(w, b, mean, std)
    assert w == pytest.approx([1 / 16, -1.4 / 16], abs=1e-6) and b == pytest.approx(0.1, abs=1e-6)
    assert history[-1] < 1e-12


def test_normal_density_and_log_density():
    assert normal_density(0.3, 0.3, 0.1) == pytest.approx(3.989, abs=1e-3)  # una densidad puede pasar de 1
    for x in (0.1, 0.3, 0.55):
        expected = torch.distributions.Normal(0.3, 0.1).log_prob(torch.tensor(x)).item()
        assert normal_log_density(x, 0.3, 0.1) == pytest.approx(expected, abs=1e-5)
        assert np.log(normal_density(x, 0.3, 0.1)) == pytest.approx(normal_log_density(x, 0.3, 0.1))


def test_sigmoid_and_gail_reward():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(2.0) == pytest.approx(0.881, abs=1e-3)
    assert gail_reward(np.array([0.1, 0.5, 0.9])) == pytest.approx([0.105, 0.693, 2.303], abs=1e-3)


def test_a_long_product_of_probabilities_underflows_in_float32():
    assert np.prod(np.full(600, 0.5, dtype=np.float32)) == 0.0
    assert np.sum(np.log(np.full(600, 0.5))) == pytest.approx(-415.888, abs=1e-3)


# --- Soluciones de los ejercicios del capítulo 0 (apéndice B) ---------------------------------------------

def test_exercise_second_step():
    w = np.array([0.05125])
    grad_w, _ = mse_gradient(w, 0.0, X, Y)
    assert grad_w[0] == pytest.approx(-1.845)
    w = w - 0.005 * grad_w
    assert w[0] == pytest.approx(0.060475)
    assert mse(predict(w, 0.0, X), Y) == pytest.approx(0.00033625125)


def test_exercise_largest_learning_rate():
    # Cada paso multiplica la distancia al óptimo por (1 − α × 164): se dispara si α > 2 / 164.
    curvature = 2 * np.mean(X[:, 0] ** 2)
    assert curvature == 164
    for learning_rate, grows in ((0.012, False), (0.0125, True), (0.02, True)):
        w = np.array([0.0])
        for _ in range(50):
            w = w - learning_rate * mse_gradient(w, 0.0, X, Y)[0]
        assert (abs(w[0] - 1 / 16) > 1 / 16) == grows


def test_exercise_log_density_and_ratio():
    old = normal_log_density(0.5, 0.3, 0.1)
    new = normal_log_density(0.5, 0.35, 0.1)
    assert old == pytest.approx(-0.616354, abs=1e-6)
    assert new == pytest.approx(0.258646, abs=1e-6)
    assert np.exp(new - old) == pytest.approx(2.3989, abs=1e-4)


def test_chapter_0_code_exercise():
    # El ejercicio de código: con 0,012 el descenso converge; con 0,0125, apenas pasado 2/164, se dispara.
    _, _, converges = gradient_descent(X, Y, 0.012, steps=200)
    _, _, diverges = gradient_descent(X, Y, 0.0125, steps=200)
    assert converges[-1] < 1e-5
    assert diverges[-1] > 1e6
