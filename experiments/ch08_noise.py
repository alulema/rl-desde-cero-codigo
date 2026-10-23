"""Capítulo 8: el entrenamiento dice que da vueltas, la evaluación dice que no.

En `ch08_debugging.py`, las policies entrenadas con la observación rellena con ceros completan casi todos
los episodios del final del entrenamiento, y casi ninguno en la evaluación. La diferencia está en cómo se
elige la acción: al entrenar, PPO la muestrea de una normal, y el entorno la recorta a su rango; al evaluar,
el agente da la media, sin ruido.

Este experimento vuelve a entrenar las mismas policies (mismas semillas, mismos pasos: salen idénticas) y
evalúa cada una de tres maneras, con los mismos 20 arranques con ruido:

- `mean`: la media de la normal, como en todas las tablas del libro;
- `sampled`: una muestra de la normal, recortada por el entorno, como al entrenar;
- `clipped_mean`: sin azar, el valor esperado de la acción ya recortada. Si con esto basta, lo que le
  faltaba a la media no era el ruido sino lo que el recorte hace con él.

    uv run python experiments/ch08_noise.py --seeds 0 1 2 3 4
"""

import argparse
from functools import partial

import numpy as np
import torch

from common import parse_args, run_metadata, save_json
from ch08_debugging import EVAL_SEEDS, SCENARIOS, make_env
from algos.ppo import PPOConfig, train
from rlcar.evaluate import FunctionPolicy, evaluate, summarize


class NoisyPolicy:
    """La policy de PPO con su ruido. `mode` es `sampled` o `clipped_mean`."""

    def __init__(self, agent, low, high, mode: str, seed: int = 0):
        self.agent, self.mode = agent, mode
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self) -> None:
        pass

    def act(self, observation, env):
        obs = self.agent.normalizer(observation) if self.agent.normalizer else observation
        with torch.no_grad():
            dist = self.agent.model.distribution(torch.as_tensor(obs, dtype=torch.float32))
            mean, std = dist.mean, dist.stddev
            if self.mode == "sampled":
                return torch.normal(mean, std, generator=self.generator).numpy()
            return clipped_mean(mean, std, self.low, self.high).numpy()


def clipped_mean(mean, std, low, high):
    """E[clip(X, low, high)] para X ~ N(mean, std²): lo que vale la acción, en promedio, después de que el
    entorno la recorta."""
    normal = torch.distributions.Normal(0.0, 1.0)
    a, b = (low - mean) / std, (high - mean) / std
    pdf = lambda z: normal.log_prob(z).exp()
    inside = mean * (normal.cdf(b) - normal.cdf(a)) + std * (pdf(a) - pdf(b))
    return low * normal.cdf(a) + high * (1 - normal.cdf(b)) + inside


def describe(result: dict) -> str:
    return (f"{result['completed_laps']}/{result['episodes']} vueltas, fracción {result['lap_fraction_mean']:.2f}, "
            f"acelerador {result['mean_throttle']:.2f}, freno {result['mean_brake']:.3f}, "
            f"{result['mean_speed']:.1f} m/s, {result['end_reasons']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=["clean", "zero_observation", "all_bugs"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser)

    results = {"metadata": run_metadata(args.seeds), "ppo_config": PPOConfig(total_steps=args.steps),
               "eval_seeds": list(EVAL_SEEDS), "scenarios": {}}
    for name in args.scenarios:
        config = SCENARIOS[name][0]()
        env = make_env(name)
        low, high = env.action_space.low, env.action_space.high
        runs = []
        for seed in args.seeds:
            agent, history = train(partial(make_env, name), PPOConfig(total_steps=args.steps, seed=seed),
                                   log=lambda row: None)
            run = {"seed": seed, "train_seconds": history[-1]["seconds"], "policy_std": history[-1]["std"]}
            for mode in ("mean", "sampled", "clipped_mean"):
                policy = (FunctionPolicy(agent.act) if mode == "mean"
                          else NoisyPolicy(agent, low, high, mode, seed=seed))
                records = evaluate(config, policy, EVAL_SEEDS)
                run[mode] = summarize(records)
                run[mode]["mean_throttle"] = float(np.mean([r.mean_throttle for r in records]))
                print(f"[{name}] semilla {seed}, {mode}: {describe(run[mode])}", flush=True)
            runs.append(run)
        results["scenarios"][name] = {"runs": runs}
        save_json("ch08_noise.json", results)

    print(f"guardado en {save_json('ch08_noise.json', results)}")


if __name__ == "__main__":
    main()
