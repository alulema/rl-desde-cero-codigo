"""Returns, ventajas y GAE "a mano", en NumPy.

El ejemplo resuelto del libro (capítulos 1 y 2) es un episodio de 4 pasos: el auto avanza tres pasos
(recompensa +1 cada uno) y en el cuarto se sale de la pista (−1, fin del episodio). El crítico estima
V = [2.0, 1.0, 0.5, −0.5]. Con γ = 0.9 y λ = 0.8, los tests de `tests/test_returns_gae.py` comprueban cada
número que el libro calcula a mano.
"""

import numpy as np


# region: discounted_return
def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """Return con descuento desde el primer paso:
    G_0 = r_0 + γ r_1 + γ² r_2 + ..."""
    weights = gamma ** np.arange(len(rewards))
    return float(np.sum(weights * rewards))
# endregion: discounted_return


# region: returns_to_go
def returns_to_go(rewards: np.ndarray, gamma: float,
                  bootstrap: float = 0.0) -> np.ndarray:
    """G_t para cada paso, de atrás hacia adelante: G_t = r_t + γ G_{t+1}.

    `bootstrap` es el valor estimado del estado donde se cortó el episodio:
    0 si terminó de verdad, V(s_T) si se cortó por tiempo.
    """
    returns = np.zeros(len(rewards))
    future = bootstrap
    for t in reversed(range(len(rewards))):
        future = rewards[t] + gamma * future
        returns[t] = future
    return returns
# endregion: returns_to_go


# region: td_errors
def td_errors(rewards: np.ndarray, values: np.ndarray, last_value: float,
              gamma: float) -> np.ndarray:
    """δ_t = r_t + γ V(s_{t+1}) − V(s_t): cuánto mejor salió el paso de lo
    que el crítico esperaba.

    r_t + γ V(s_{t+1}) es una estimación de Q(s_t, a_t), así que δ_t es
    también una estimación de la ventaja A(s_t, a_t) = Q − V.
    """
    next_values = np.append(values[1:], last_value)
    return rewards + gamma * next_values - values
# endregion: td_errors


# region: gae
def gae(rewards: np.ndarray, values: np.ndarray, last_value: float,
        gamma: float, lam: float) -> np.ndarray:
    """Generalized Advantage Estimation de un tramo de un solo episodio:
    A_t = δ_t + γλ A_{t+1}.

    Con λ = 0, A_t = δ_t (poca varianza, mucho sesgo si el crítico se
    equivoca). Con λ = 1, A_t = G_t − V(s_t) (sin sesgo del crítico, mucha
    varianza).
    """
    deltas = td_errors(rewards, values, last_value, gamma)
    advantages = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * lam * running
        advantages[t] = running
    return advantages
# endregion: gae


# region: gae_batch
def gae_batch(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ended: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """GAE para lo que junta PPO: arreglos (pasos, entornos), con episodios que terminan a la mitad.

    - `next_values[t]` es V del estado que siguió al paso t: el del mismo episodio, aunque el entorno ya se
      haya reiniciado (por eso se usa `final_obs`).
    - `terminated[t]`: el episodio terminó de verdad. No hay futuro: no se suma γ V(s_{t+1}).
    - `episode_ended[t]`: terminó o se cortó por tiempo. La recursión de GAE no cruza al episodio siguiente,
      pero si fue un corte por tiempo sí se usa V(s_{t+1}), porque el futuro existía.
    """
    not_terminated = 1.0 - terminated
    continues = 1.0 - episode_ended
    deltas = rewards + gamma * next_values * not_terminated - values
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1:])
    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * lam * continues[t] * running
        advantages[t] = running
    return advantages
# endregion: gae_batch
