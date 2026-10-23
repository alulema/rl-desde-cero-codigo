"""Combinar imitación y RL (capítulo 5): BC como pérdida auxiliar dentro de PPO.

En la variante simultánea, cada lote de PPO se entrena como siempre (con la recompensa del entorno y, si
se quiere, la de GAIL) y después la policy da unos pasos de BC sobre las demostraciones: la media del
actor se acerca a la acción del experto. Se sigue el módulo de BC de ML-Agents con la configuración que
se preparó para Agentic Racing (`strength: 0.5`, `steps: 2000000` de 10 millones): su propio optimizador,
con la mitad de la tasa de aprendizaje de PPO, que baja linealmente y se apaga al 20 % del entrenamiento.

Simplificaciones: la pérdida es el error cuadrático de la media del actor, como `algos.bc`, sumado sobre
las tres acciones (ML-Agents compara la acción muestreada y promedia); y cada lote da 3 × 8 pasos de 256
pares (con la configuración de Agentic Racing, ML-Agents daría unos 6 pasos de 2 048).
"""

from dataclasses import dataclass

import numpy as np
import torch

from algos.bc import Demonstrations


@dataclass(frozen=True)
class BCAuxConfig:
    strength: float = 0.5  # tasa de aprendizaje de BC = strength × la de PPO
    active_fraction: float = 0.2  # 2 de 10 millones de decisiones en Agentic Racing
    epochs: int = 3
    minibatches_per_epoch: int = 8  # los mismos que el PPO del libro: 2 048 / 256
    minibatch_size: int = 256
    seed: int = 0


class BCAuxLoss:
    """El `update_hook` de `algos.ppo.train` para la variante simultánea."""

    def __init__(self, demos: Demonstrations, ppo_learning_rate: float, config: BCAuxConfig):
        self.demos, self.config = demos, config
        self.learning_rate = config.strength * ppo_learning_rate
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = None

    # region: bc_aux
    def __call__(self, model, normalize, progress: float) -> dict:
        """Unos pasos de BC después de cada lote de PPO."""
        cfg = self.config
        if progress >= cfg.active_fraction:
            return {"bc_loss": float("nan"), "bc_learning_rate": 0.0}
        if self.optimizer is None:  # su propio optimizador, no el de PPO
            self.optimizer = torch.optim.Adam(model.actor.parameters(),
                                              lr=self.learning_rate)
        lr = self.learning_rate * (1 - progress / cfg.active_fraction)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        # El normalizador de PPO, el mismo con el que actúa la policy.
        obs = normalize(self.demos.observations)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        target = torch.as_tensor(self.demos.actions, dtype=torch.float32)
        losses = []
        for _ in range(cfg.epochs * cfg.minibatches_per_epoch):
            idx = self.rng.integers(len(target), size=cfg.minibatch_size)
            idx = torch.as_tensor(idx)
            error = model.actor(obs[idx]) - target[idx]
            loss = (error ** 2).sum(dim=1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return {"bc_loss": float(np.mean(losses)), "bc_learning_rate": lr}
    # endregion: bc_aux
