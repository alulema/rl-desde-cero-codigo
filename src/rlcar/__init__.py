"""Entorno de juguete del libro: un auto 2D en una pista, con la API de Gymnasium.

Es una versión mínima de Agentic Racing: ver docs/rlcar-design.md.
"""

import gymnasium as gym

from rlcar import presets
from rlcar.env import EnvConfig, RLCarEnv

gym.register(id="rlcar/RLCar-v0", entry_point="rlcar.env:RLCarEnv")

__all__ = ["EnvConfig", "RLCarEnv", "presets"]
