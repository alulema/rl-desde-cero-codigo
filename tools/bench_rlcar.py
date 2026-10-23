"""Mide cuántas decisiones por segundo da `rlcar`, con uno y con varios entornos en paralelo.

    uv run python tools/bench_rlcar.py --envs 1 8 --seconds 10
"""

import argparse
import os
import platform
import time

import gymnasium as gym
import numpy as np

from rlcar import RLCarEnv, presets


def make_env():
    return RLCarEnv(presets.agentic())


def bench(n_envs: int, vector: str, seconds: float) -> float:
    if n_envs == 1:
        env = make_env()
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        steps, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            _, _, terminated, truncated, _ = env.step(rng.uniform([-0.2, 0.2, 0], [0.2, 1, 0.1]))
            steps += 1
            if terminated or truncated:
                env.reset()
        return steps / (time.perf_counter() - start)

    cls = gym.vector.AsyncVectorEnv if vector == "async" else gym.vector.SyncVectorEnv
    envs = cls([make_env] * n_envs)
    envs.reset(seed=0)
    rng = np.random.default_rng(0)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        envs.step(rng.uniform([-0.2, 0.2, 0], [0.2, 1, 0.1], size=(n_envs, 3)))
        steps += n_envs
    rate = steps / (time.perf_counter() - start)
    envs.close()
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(f"{platform.platform()} · {platform.processor() or platform.machine()} · {os.cpu_count()} CPUs")
    for n in args.envs:
        for vector in (["sync"] if n == 1 else ["sync", "async"]):
            print(f"{n} entorno(s), {vector:5s}: {bench(n, vector, args.seconds):8.0f} decisiones/s")


if __name__ == "__main__":
    main()
