import numpy as np
import pytest

from algos.bc import BCConfig, Demonstrations, rollout, train_bc
from algos.dagger import DAggerConfig, dagger
from rlcar import presets
from rlcar.expert import Expert


def test_bc_learns_a_linear_map():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(2_000, 4))
    actions = np.tanh(obs[:, :3] * 0.5)
    policy = train_bc(Demonstrations(obs, actions), BCConfig(gradient_steps=2_000))
    assert policy.loss(Demonstrations(obs, actions)) < 0.01


def test_expert_driving_has_zero_disagreement_and_laps():
    r = rollout(presets.technical(), Expert(memory=True), seed=3)
    assert r.end_reason == "lap"
    assert len(r.demonstrations) == len(r.driven_actions) == len(r.lateral)
    assert np.all(r.disagreement == 0)


def test_shadow_expert_follows_the_real_trajectory():
    """Con la policy manejando, el experto con memoria se consulta en cada paso: su volante filtrado sigue
    sus propias etiquetas, así que dos corridas iguales dan las mismas etiquetas."""
    config = presets.technical()
    driver = lambda obs: np.array([0.0, 0.5, 0.0])
    a = rollout(config, Expert(memory=True), seed=1, driver=driver)
    b = rollout(config, Expert(memory=True), seed=1, driver=driver)
    np.testing.assert_array_equal(a.demonstrations.actions, b.demonstrations.actions)
    assert np.all(a.driven_actions == [0.0, 0.5, 0.0])


def test_dagger_aggregates_policy_states():
    config = presets.technical()
    demos = rollout(config, Expert(memory=False), seed=0).demonstrations
    sizes = []
    dagger(config, Expert(memory=False), demos,
           DAggerConfig(iterations=1, episodes_per_iteration=1, bc=BCConfig(gradient_steps=50)),
           lambda i: [10 + i], lambda i, policy, rollouts, data: sizes.append(len(data)))
    assert sizes[0] == len(demos) and sizes[1] > sizes[0]


def bc_loss(predicted, expert):
    """La pérdida de BC del libro: error cuadrático sumado sobre las 3 acciones, promediado sobre los
    ejemplos (como `train_bc` y `BCPolicy.loss`)."""
    return float(((np.asarray(predicted) - np.asarray(expert)) ** 2).sum(axis=1).mean())


def test_chapter_3_bc_loss_by_hand():
    expert = [[-0.35, 0.72, 0.0], [0.10, 0.0, 0.6]]
    policy = [[-0.25, 0.80, 0.05], [0.10, 0.30, 0.20]]
    assert bc_loss(policy[:1], expert[:1]) == pytest.approx(0.0189)
    assert bc_loss(policy[1:], expert[1:]) == pytest.approx(0.25)
    assert bc_loss(policy, expert) == pytest.approx(0.13445)


def test_chapter_3_squared_error_averages_contradicting_labels():
    """Dos etiquetas opuestas para la misma observación: la mejor predicción constante es el promedio, y
    en la física de `rlcar` ese promedio frena."""
    from rlcar.physics import CarParams, drive_acceleration

    labels = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.8]])
    mean = labels.mean(axis=0)
    np.testing.assert_allclose(mean, [0.0, 0.5, 0.4])
    assert bc_loss([mean] * 2, labels) == pytest.approx(0.41)
    assert bc_loss([labels[0]] * 2, labels) == pytest.approx(0.82)
    # Probar alrededor del promedio: ninguna constante cercana da menos.
    for delta in ([0, 0.05, 0], [0, -0.05, 0], [0, 0, 0.05], [0, 0, -0.05]):
        assert bc_loss([mean + delta] * 2, labels) > 0.41
    params = CarParams()
    assert drive_acceleration(20.0, 0.5, 0.4, params) == pytest.approx(-6.0)
    assert drive_acceleration(20.0, 1.0, 0.0, params) > 0


def test_chapter_3_exercises():
    # 1. La pérdida a mano.
    expert = [[0.2, 1.0, 0.0], [-0.5, 0.0, 0.5]]
    policy = [[0.0, 0.9, 0.1], [-0.4, 0.2, 0.3]]
    assert bc_loss(policy[:1], expert[:1]) == pytest.approx(0.06)
    assert bc_loss(policy[1:], expert[1:]) == pytest.approx(0.09)
    assert bc_loss(policy, expert) == pytest.approx(0.075)
    # 2. Tres etiquetas para la misma observación: el promedio, y lo que hace el auto.
    from rlcar.physics import CarParams, drive_acceleration

    labels = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.9]])
    mean = labels.mean(axis=0)
    np.testing.assert_allclose(mean, [0.0, 2 / 3, 0.3])
    assert drive_acceleration(20.0, mean[1], mean[2], CarParams()) == pytest.approx(-4.5)
    # 3. Épocas: 5.000 pasos × 256 ejemplos, con 300 pares y con 10.000.
    config = BCConfig()
    seen = config.gradient_steps * config.minibatch_size
    assert seen / 300 == pytest.approx(4266.67, abs=0.01)
    assert seen / 10_000 == pytest.approx(128)
    # 4. Ross y Bagnell: de 300 a 800 decisiones, T crece 2,67 veces y T² crece 7,11.
    assert 800 / 300 == pytest.approx(2.667, abs=0.001)
    assert (800 / 300) ** 2 == pytest.approx(7.111, abs=0.001)
