"""GAIL (Ho y Ermon 2016), capítulo 4: un discriminador como recompensa de imitación.

El discriminador D(s, a) es un clasificador: da la probabilidad de que el par (observación, acción) venga
del experto. Se entrena con los pares del experto (etiqueta 1) contra los del agente (etiqueta 0). El
agente recibe la recompensa −log(1 − D): alta cuando el par se parece a los del experto.

Esa recompensa es **siempre positiva**. Cada paso vivo paga, así que terminar el episodio (salirse,
detenerse o completar la vuelta) corta el pago futuro: es el sesgo de supervivencia.

Se sigue el discriminador de ML-Agents 1.1.0, con el que se configuró Agentic Racing: 2 capas de 128,
observación y acción como entrada (`use_actions: true`), penalización de gradiente con peso 10. Tres
simplificaciones:
- no se le pasa la marca de fin de episodio;
- las dos recompensas comparten un solo crítico y γ = 0.995 (ML-Agents estima un valor por cada señal, y
  la configuración de Agentic Racing le daba a GAIL γ = 0.99);
- la penalización de gradiente se aplica al puntaje antes de la sigmoide, respecto de la observación y la
  acción, con una sola mezcla por par. ML-Agents la aplica a D, solo respecto de la observación, y mezcla
  elemento por elemento.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from algos.bc import Demonstrations
from algos.ppo import RunningNormalizer, mlp


@dataclass(frozen=True)
class GAILConfig:
    gail_weight: float = 0.15  # `strength` de GAIL en la configuración de Agentic Racing
    env_weight: float = 1.0  # `strength` de la recompensa del entorno
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    epochs: int = 3  # pasadas por lote de PPO, como `num_epoch` de Agentic Racing
    minibatch_size: int = 256
    gradient_penalty: float = 10.0
    seed: int = 0


# region: gail_reward
def gail_reward(d: np.ndarray) -> np.ndarray:
    """−log(1 − D). D = 0.1 → 0.105; D = 0.5 → 0.693; D = 0.9 → 2.303.
    El 1e-7 evita el infinito si D llega a 1."""
    return -np.log(1.0 - d + 1e-7)
# endregion: gail_reward


class GAIL:
    """La recompensa de GAIL como `reward_hook` de `algos.ppo.train`."""

    def __init__(self, demos: Demonstrations, action_low: np.ndarray, action_high: np.ndarray, config: GAILConfig):
        torch.manual_seed(config.seed)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_low, self.action_high = action_low, action_high
        # Las observaciones se normalizan con la media y la varianza de las demostraciones, fijas.
        self.normalizer = RunningNormalizer(demos.observations.shape[1])
        self.normalizer.update(demos.observations)
        self.expert = self._inputs(demos.observations, demos.actions)
        self.discriminator = mlp(self.expert.shape[1], config.hidden_sizes, 1, output_gain=1.0)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.learning_rate)

    def _inputs(self, observations: np.ndarray, actions: np.ndarray) -> torch.Tensor:
        # PPO guarda la acción muestreada, sin recortar; el entorno ejecutó la recortada. El discriminador ve
        # la ejecutada: si no, distinguiría al agente por acciones fuera de rango.
        actions = np.clip(actions, self.action_low, self.action_high)
        return torch.as_tensor(np.concatenate([self.normalizer(observations), actions], axis=1), dtype=torch.float32)

    @torch.no_grad()
    def probability(self, inputs: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(self.discriminator(inputs).squeeze(-1)).numpy()

    # region: discriminator_loss
    def discriminator_loss(self, expert: torch.Tensor,
                           agent: torch.Tensor) -> torch.Tensor:
        """Clasificación binaria (experto = 1, agente = 0) más la
        penalización de gradiente: en puntos intermedios entre pares del
        experto y del agente, la norma del gradiente del puntaje (antes de
        la sigmoide) tiene que estar cerca de 1, para que D no se vuelva
        tan tajante que deje de dar una señal útil."""
        logits_expert = self.discriminator(expert).squeeze(-1)
        logits_agent = self.discriminator(agent).squeeze(-1)
        bce = nn.functional.binary_cross_entropy_with_logits
        loss = (bce(logits_expert, torch.ones_like(logits_expert))
                + bce(logits_agent, torch.zeros_like(logits_agent)))
        mix = torch.rand(len(expert), 1)
        between = (mix * expert + (1 - mix) * agent).requires_grad_(True)
        grad, = torch.autograd.grad(self.discriminator(between).sum(),
                                    between, create_graph=True)
        penalty = ((grad.norm(dim=1) - 1) ** 2).mean()
        return loss + self.config.gradient_penalty * penalty
    # endregion: discriminator_loss

    def update(self, agent: torch.Tensor) -> float:
        losses = []
        for _ in range(self.config.epochs):
            for idx in torch.randperm(len(agent)).split(self.config.minibatch_size):
                expert = self.expert[torch.as_tensor(self.rng.integers(len(self.expert), size=len(idx)))]
                loss = self.discriminator_loss(expert, agent[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    # region: gail_hook
    def __call__(self, raw_obs: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray):
        """Recompensa del lote con el discriminador actual; después, el
        discriminador aprende del lote."""
        shape = rewards.shape
        agent = self._inputs(raw_obs.reshape(-1, raw_obs.shape[-1]),
                             actions.reshape(-1, actions.shape[-1]))
        d_agent = self.probability(agent)
        imitation = gail_reward(d_agent).reshape(shape)
        idx = self.rng.integers(len(self.expert), size=len(agent))
        d_expert = self.probability(self.expert[torch.as_tensor(idx)])
        loss = self.update(agent)
        stats = {"d_expert": float(d_expert.mean()),
                 "d_agent": float(d_agent.mean()),
                 "gail_reward": float(imitation.mean()),
                 "env_reward": float(rewards.mean()),
                 "discriminator_loss": loss}
        mixed = (self.config.env_weight * rewards
                 + self.config.gail_weight * imitation)
        return mixed, stats
    # endregion: gail_hook
