"""Capítulo 2: el mismo experimento con el PPO de Stable-Baselines3, como referencia.

Mismos hiperparámetros que `algos.ppo.PPOConfig` (8 entornos, 256 pasos por entorno, 10 épocas, minibatch
de 256, γ = 0.995, λ = 0.95, ε = 0.2, red de 64 × 64, tasa de aprendizaje que baja linealmente a 0 y
observaciones normalizadas). Si nuestro PPO y el de SB3 dan resultados parecidos, el nuestro no tiene un
bug grosero.

    uv run python experiments/ch02_sb3_baseline.py --circuit easy --seeds 0 1 2 --steps 500000
"""

import argparse
import time
from functools import partial

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from common import parse_args, run_metadata, save_json
from rlcar import RLCarEnv, presets
from rlcar.evaluate import FunctionPolicy, evaluate, summarize

EVAL_SEEDS = range(1000, 1010)
CIRCUITS = {"easy": presets.agentic, "technical": presets.technical}


def make_env(circuit: str):
    return RLCarEnv(CIRCUITS[circuit]())


def train_sb3(circuit: str, seed: int, steps: int):
    envs = VecNormalize(SubprocVecEnv([partial(make_env, circuit)] * 8), norm_obs=True, norm_reward=False)
    envs.seed(seed)
    model = PPO("MlpPolicy", envs, n_steps=256, batch_size=256, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, learning_rate=lambda progress: 3e-4 * progress, ent_coef=0.0, vf_coef=0.5,
                max_grad_norm=0.5, policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0)
    start = time.perf_counter()
    model.learn(total_timesteps=steps)
    seconds = time.perf_counter() - start
    envs.training = False

    def act(observation):
        return model.predict(envs.normalize_obs(observation), deterministic=True)[0]

    summary = summarize(evaluate(CIRCUITS[circuit](), FunctionPolicy(act), EVAL_SEEDS))
    envs.close()
    return seconds, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), default="easy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser, selectors=('circuit',))
    results = {"metadata": run_metadata(args.seeds), "circuit": args.circuit,
               "stable_baselines3": stable_baselines3.__version__, "steps": args.steps, "runs": []}
    for seed in args.seeds:
        seconds, summary = train_sb3(args.circuit, seed, args.steps)
        results["runs"].append({"seed": seed, "train_seconds": round(seconds, 1), "eval": summary})
        lap_time = f"{summary['lap_time_mean']:.1f} s" if summary["lap_time_mean"] else "—"
        print(f"SB3 semilla {seed}: {seconds:.0f} s | eval: vuelta {summary['lap_fraction_mean']:.3f}, "
              f"{summary['completed_laps']}/{summary['episodes']} vueltas, {lap_time}, "
              f"freno {summary['mean_brake']:.3f}, fines {summary['end_reasons']}")
    print(f"guardado en {save_json(f'ch02_sb3_{args.circuit}.json', results)}")


if __name__ == "__main__":
    main()
