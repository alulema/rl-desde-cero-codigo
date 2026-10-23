"""Behavioral cloning (capítulo 3): aprendizaje supervisado sobre las demostraciones del experto.

Las demostraciones son pares (observación, acción del experto). La policy es una red que, dada la
observación, predice la acción, y se entrena minimizando el error cuadrático contra la acción del experto.
No hay recompensa ni entorno durante el entrenamiento: el entorno solo se usa para grabar y para evaluar.

El experto lee el estado privilegiado (`env.state`); la policy, solo la observación. Si el experto usa
información que la observación no tiene (por ejemplo, su volante filtrado), la policy no puede copiarlo
exactamente.
"""

from dataclasses import dataclass

import numpy as np
import torch

from algos.ppo import RunningNormalizer, mlp
from rlcar.env import EnvConfig, RLCarEnv
from rlcar.expert import Expert


@dataclass(frozen=True)
class BCConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)  # la misma red que el actor de PPO
    learning_rate: float = 1e-3
    gradient_steps: int = 5_000  # fijo, no por épocas: así DAgger, con más datos, no entrena más pasos
    minibatch_size: int = 256
    seed: int = 0


@dataclass
class Demonstrations:
    observations: np.ndarray
    actions: np.ndarray  # la acción del experto en cada observación

    def __len__(self) -> int:
        return len(self.observations)

    def __add__(self, other: "Demonstrations") -> "Demonstrations":
        return Demonstrations(np.concatenate([self.observations, other.observations]),
                              np.concatenate([self.actions, other.actions]))


@dataclass
class Rollout:
    """Un episodio grabado. Con `driver=None` maneja el experto; si no, maneja la policy y el experto solo
    dice qué habría hecho en cada paso."""

    demonstrations: Demonstrations
    driven_actions: np.ndarray  # lo que se ejecutó de verdad
    lateral: np.ndarray  # desplazamiento lateral (m) después de cada decisión
    end_reason: str
    lap_fraction: float
    duration: float

    @property
    def disagreement(self) -> np.ndarray:
        """Error cuadrático, paso por paso, entre lo que se hizo y lo que el experto habría hecho."""
        return ((self.driven_actions - self.demonstrations.actions) ** 2).sum(axis=1)


# region: rollout
def rollout(config: EnvConfig, expert: Expert, seed: int,
            driver=None) -> Rollout:
    """Graba un episodio. El experto se consulta en **cada** paso, también
    cuando maneja la policy: si tiene memoria (volante filtrado,
    temporizadores), tiene que seguir la trayectoria real, no una consulta
    suelta (DAgger, §6.3)."""
    env = RLCarEnv(config)
    observation, _ = env.reset(seed=seed)
    expert.reset()
    low, high = env.action_space.low, env.action_space.high
    observations, labels, driven, lateral = [], [], [], []
    while True:
        label = expert.act(env.state)
        if driver is None:
            action = label
        else:
            action = np.asarray(driver(observation), dtype=float)
        action = np.clip(action, low, high)
        observations.append(observation)
        labels.append(np.clip(label, low, high))
        driven.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        lateral.append(env.state.projection.lateral)
        if terminated or truncated:
            break
    lap_fraction = float(min(1.0, max(0.0, env.state.lap_fraction)))
    demos = Demonstrations(np.array(observations), np.array(labels))
    return Rollout(demos, np.array(driven), np.array(lateral),
                   info["end_reason"], lap_fraction, env.state.time)
# endregion: rollout


def record_demonstrations(config: EnvConfig, expert: Expert, seeds) -> Demonstrations:
    rollouts = [rollout(config, expert, seed) for seed in seeds]
    demos = rollouts[0].demonstrations
    for r in rollouts[1:]:
        demos = demos + r.demonstrations
    return demos


class BCPolicy:
    def __init__(self, model: torch.nn.Module, normalizer: RunningNormalizer):
        self.model = model
        self.normalizer = normalizer

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(self.normalizer(observation), dtype=torch.float32)
        return self.model(obs).numpy().astype(float)

    @torch.no_grad()
    def loss(self, demos: Demonstrations) -> float:
        """El error cuadrático medio (sumado sobre las 3 acciones) en un conjunto de demostraciones."""
        obs = torch.as_tensor(self.normalizer(demos.observations), dtype=torch.float32)
        target = torch.as_tensor(demos.actions, dtype=torch.float32)
        return float(((self.model(obs) - target) ** 2).sum(dim=1).mean())


# region: train_bc
def train_bc(demos: Demonstrations, config: BCConfig) -> BCPolicy:
    """Regresión de la acción del experto desde la observación, con error
    cuadrático."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    n_obs, n_actions = demos.observations.shape[1], demos.actions.shape[1]
    # Media y varianza de los datos, fijas durante el entrenamiento.
    normalizer = RunningNormalizer(n_obs)
    normalizer.update(demos.observations)
    model = mlp(n_obs, config.hidden_sizes, n_actions, output_gain=0.01)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    obs = torch.as_tensor(normalizer(demos.observations),
                          dtype=torch.float32)
    target = torch.as_tensor(demos.actions, dtype=torch.float32)
    for _ in range(config.gradient_steps):
        idx = rng.integers(len(demos), size=config.minibatch_size)
        batch = torch.as_tensor(idx)
        error = model(obs[batch]) - target[batch]
        loss = (error ** 2).sum(dim=1).mean()  # suma de las 3 acciones
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return BCPolicy(model, normalizer)
# endregion: train_bc
