"""PPO mínimo en PyTorch, en un solo archivo (capítulo 2).

El ciclo es el de siempre:
1. Junta `n_steps` decisiones en cada uno de los `n_envs` entornos con la policy actual (on-policy).
2. Calcula las ventajas con GAE (`algos.returns_gae.gae_batch`).
3. Mejora la policy unas cuantas épocas sobre ese lote, con el objetivo recortado (clipping) para no
   alejarse demasiado de la policy que juntó los datos. Después, el lote se tira.

Actor y crítico son redes separadas (como en ML-Agents). La policy es una normal por cada dimensión de
la acción: la red da la media y la desviación estándar es un parámetro aprendido aparte. El entorno
recorta la acción a su rango.
"""

import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from algos.returns_gae import gae_batch


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 1_000_000  # decisiones en total, sumando todos los entornos
    n_envs: int = 8
    n_steps: int = 256  # decisiones por entorno en cada lote: 8 × 256 = 2 048 por lote
    epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.995  # el de Agentic Racing
    lam: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4  # baja linealmente hasta 0
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    initial_log_std: float = 0.0
    normalize_observations: bool = True
    normalize_advantages: bool = True
    seed: int = 0


# region: networks
def mlp(input_size: int, hidden_sizes: tuple[int, ...], output_size: int, output_gain: float) -> nn.Sequential:
    layers, size = [], input_size
    for hidden in hidden_sizes:
        linear = nn.Linear(size, hidden)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        size = hidden
    output = nn.Linear(size, output_size)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int, config: PPOConfig):
        super().__init__()
        # Ganancia chica en la salida del actor: al principio todas las medias están cerca de 0.
        self.actor = mlp(obs_size, config.hidden_sizes, action_size, output_gain=0.01)
        self.critic = mlp(obs_size, config.hidden_sizes, 1, output_gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_size,), config.initial_log_std))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
# endregion: networks


# region: running_normalizer
class RunningNormalizer:
    """Media y varianza móviles de las observaciones, para que todas
    lleguen a la red en escalas parecidas."""

    def __init__(self, size: int):
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)
        n = len(batch)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + batch_var * n
                    + delta**2 * self.count * n / total) / total
        self.count = total

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        scaled = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(scaled, -10, 10)
# endregion: running_normalizer


class EpisodeTracker:
    """Lleva el return y la duración de cada episodio en curso, y guarda los que terminan."""

    def __init__(self, n_envs: int):
        self.returns = np.zeros(n_envs)
        self.lengths = np.zeros(n_envs, dtype=int)
        self.finished: list[dict] = []

    def step(self, rewards: np.ndarray, ended: np.ndarray, info: dict) -> None:
        self.returns += rewards
        self.lengths += 1
        for i in np.flatnonzero(ended):
            final = info["final_info"]
            self.finished.append({
                "return": float(self.returns[i]),
                "length": int(self.lengths[i]),
                "lap_fraction": float(final["lap_fraction"][i]),
                "end_reason": str(final["end_reason"][i]),
            })
            self.returns[i], self.lengths[i] = 0.0, 0

    def pop(self) -> list[dict]:
        finished, self.finished = self.finished, []
        return finished


# region: clipped_objective
def clipped_objective(ratio: torch.Tensor, advantages: torch.Tensor,
                      epsilon: float) -> torch.Tensor:
    """min(r·A, clip(r, 1−ε, 1+ε)·A), paso por paso.

    Si la acción fue mejor que lo esperado (A > 0), subir su probabilidad
    paga hasta r = 1 + ε y después no paga más. Si fue peor (A < 0), bajarla
    paga hasta r = 1 − ε. En la dirección contraria no hay tope: el min deja
    pasar el castigo completo.
    """
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    return torch.min(unclipped, clipped)
# endregion: clipped_objective


# region: ppo_loss
def ppo_loss(model: ActorCritic, obs, actions, old_log_probs, advantages,
             returns, config: PPOConfig):
    """La pérdida de PPO para un minibatch. Devuelve la pérdida total y sus
    partes, para registrarlas."""
    dist = model.distribution(obs)
    log_probs = dist.log_prob(actions).sum(-1)
    ratio = (log_probs - old_log_probs).exp()  # π_θ(a|s) / π_θ_viejo(a|s)
    objective = clipped_objective(ratio, advantages, config.clip_epsilon)
    policy_loss = -objective.mean()

    value_loss = ((model.value(obs) - returns) ** 2).mean()
    entropy = dist.entropy().sum(-1).mean()
    loss = (policy_loss + config.value_coef * value_loss
            - config.entropy_coef * entropy)

    with torch.no_grad():
        approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean()
        clipped = (ratio - 1).abs() > config.clip_epsilon
        clip_fraction = clipped.float().mean()
    stats = {"policy_loss": policy_loss.item(),
             "value_loss": value_loss.item(), "entropy": entropy.item(),
             "approx_kl": approx_kl.item(),
             "clip_fraction": clip_fraction.item()}
    return loss, stats
# endregion: ppo_loss


class PPOAgent:
    """La policy entrenada: normalizador + red. `act` da la acción media (sin ruido) para evaluar."""

    def __init__(self, model: ActorCritic, normalizer: RunningNormalizer | None):
        self.model = model
        self.normalizer = normalizer

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = self.normalizer(obs) if self.normalizer else obs
        with torch.no_grad():
            return self.model.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def save(self, path: str, config: PPOConfig) -> None:
        torch.save({"model": self.model.state_dict(), "config": asdict(config),
                    "normalizer": vars(self.normalizer) if self.normalizer else None}, path)


# region: train
def train(make_env, config: PPOConfig, log=print, vector: str = "async",
          reward_hook=None, update_hook=None, init: PPOAgent | None = None) -> tuple[PPOAgent, list[dict]]:
    """Entrena con PPO. `make_env` crea un entorno; se abren `n_envs` en paralelo.

    `reward_hook(raw_obs, actions, rewards)`, si se da, recibe el lote recién juntado y devuelve la
    recompensa con la que se entrena y un diccionario de métricas. Así se suma GAIL (capítulo 4) sin tocar
    el resto del ciclo. Las métricas de episodio (`episode_return`) siguen siendo las del entorno.

    `update_hook(model, normalize, progress)` corre después de las épocas de PPO de cada lote (`progress`
    va de 0 a 1) y devuelve métricas: así se suma la pérdida de BC (capítulo 5).

    `init` arranca desde un agente ya entrenado (red y normalizador), como `--initialize-from` de
    ML-Agents. El optimizador y la tasa de aprendizaje empiezan de cero, igual que allá.

    Devuelve el agente y una fila de métricas por lote (para las curvas del capítulo).
    """
    torch.manual_seed(config.seed)
    vector_cls = gym.vector.AsyncVectorEnv if vector == "async" else gym.vector.SyncVectorEnv
    envs = vector_cls([make_env] * config.n_envs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs_size = envs.single_observation_space.shape[0]
    action_size = envs.single_action_space.shape[0]
    model = ActorCritic(obs_size, action_size, config)
    normalizer = RunningNormalizer(obs_size) if config.normalize_observations else None
    if init is not None:
        model.load_state_dict(init.model.state_dict())
        if normalizer and init.normalizer:
            normalizer.mean, normalizer.var, normalizer.count = (init.normalizer.mean.copy(),
                                                                 init.normalizer.var.copy(), init.normalizer.count)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    normalize = normalizer if normalizer else (lambda x: x)
    tracker = EpisodeTracker(config.n_envs)

    obs, _ = envs.reset(seed=config.seed)
    batch_size = config.n_envs * config.n_steps
    n_updates = config.total_steps // batch_size
    history, start = [], time.perf_counter()
    shape = (config.n_steps, config.n_envs)

    for update in range(n_updates):
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * (1 - update / n_updates)

        # 1. Juntar datos con la policy actual. El normalizador no cambia durante el lote.
        raw_obs = np.zeros(shape + (obs_size,))
        next_raw_obs = np.zeros(shape + (obs_size,))
        actions = np.zeros(shape + (action_size,))
        log_probs, rewards = np.zeros(shape), np.zeros(shape)
        terminated, ended = np.zeros(shape), np.zeros(shape)
        for t in range(config.n_steps):
            with torch.no_grad():
                dist = model.distribution(torch.as_tensor(normalize(obs), dtype=torch.float32))
                action = dist.sample()
                log_probs[t] = dist.log_prob(action).sum(-1).numpy()
            next_obs, reward, term, trunc, info = envs.step(action.numpy())
            done = term | trunc
            raw_obs[t], actions[t], rewards[t] = obs, action.numpy(), reward
            terminated[t], ended[t] = term, done
            # Si el episodio terminó, next_obs ya es el arranque del siguiente: el estado que siguió es final_obs.
            next_raw_obs[t] = next_obs
            for i in np.flatnonzero(done):
                next_raw_obs[t, i] = info["final_obs"][i]
            tracker.step(reward, done, info)
            obs = next_obs

        hook_stats = {}
        if reward_hook:
            rewards, hook_stats = reward_hook(raw_obs, actions, rewards)

        # 2. Ventajas con GAE.
        flat = lambda x: torch.as_tensor(normalize(x.reshape(-1, obs_size)), dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            values = model.value(flat(raw_obs)).numpy().reshape(shape)
            next_values = model.value(flat(next_raw_obs)).numpy().reshape(shape)
        advantages = gae_batch(rewards, values, next_values, terminated, ended, config.gamma, config.lam)
        returns = advantages + values

        # 3. Varias épocas de minibatches sobre el mismo lote.
        b_obs = flat(raw_obs)
        b_actions = torch.as_tensor(actions.reshape(-1, action_size), dtype=torch.float32)
        b_log_probs = torch.as_tensor(log_probs.reshape(-1), dtype=torch.float32)
        b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
        epoch_stats = []
        for _ in range(config.epochs):
            for idx in torch.randperm(batch_size).split(config.minibatch_size):
                adv = b_advantages[idx]
                if config.normalize_advantages:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                loss, stats = ppo_loss(model, b_obs[idx], b_actions[idx], b_log_probs[idx], adv, b_returns[idx],
                                       config)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                epoch_stats.append(stats)

        if update_hook:
            hook_stats.update(update_hook(model, normalize, update / n_updates))

        if normalizer:
            normalizer.update(raw_obs.reshape(-1, obs_size))

        episodes = tracker.pop()
        row = {"update": update + 1, "steps": (update + 1) * batch_size,
               "seconds": round(time.perf_counter() - start, 1), "episodes": len(episodes),
               "std": model.log_std.exp().mean().item()}
        row.update({k: float(np.mean([s[k] for s in epoch_stats])) for k in epoch_stats[0]})
        row.update(hook_stats)
        if episodes:
            row["episode_return"] = float(np.mean([e["return"] for e in episodes]))
            row["episode_length"] = float(np.mean([e["length"] for e in episodes]))
            row["lap_fraction"] = float(np.mean([e["lap_fraction"] for e in episodes]))
            reasons = [e["end_reason"] for e in episodes]
            for reason in ("lap", "off_track", "stall", "stuck", "wrong_way", "time_limit"):
                row[f"end_{reason}"] = reasons.count(reason) / len(reasons)
        history.append(row)
        log(row)

    envs.close()
    return PPOAgent(model, normalizer), history
# endregion: train
