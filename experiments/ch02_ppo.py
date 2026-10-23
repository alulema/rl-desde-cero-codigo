"""Capítulo 2: PPO en los circuitos fijos, con varias semillas.

Entrena el PPO mínimo (`algos.ppo`) con la recompensa de Agentic Racing (`presets.agentic()`) en el circuito
fácil (`--circuit easy`) o en el técnico, con horquillas (`--circuit technical`). Para cada semilla guarda
las curvas de entrenamiento, el modelo y una evaluación independiente de la recompensa (la acción media de
la policy, sin ruido, en episodios con semillas que no se usaron para entrenar).

    uv run python experiments/ch02_ppo.py --circuit easy --seeds 0 1 2 3 4 --steps 500000
"""

import argparse
import csv
from dataclasses import replace
from functools import partial

import numpy as np

from common import parse_args, results_dir, run_metadata, save_json
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.evaluate import ExpertPolicy, FunctionPolicy, evaluate, summarize
from rlcar.render import episode_gif

EVAL_SEEDS = range(1000, 1010)  # distintas de las semillas de entrenamiento
CIRCUITS = {"easy": presets.agentic, "technical": presets.technical}


def make_env(circuit: str = "easy"):
    return RLCarEnv(CIRCUITS[circuit]())


def save_curves(path, history):
    fields = list(dict.fromkeys(key for row in history for key in row))  # en orden de aparición
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def first_update_with_all_laps(history) -> int | None:
    """Los pasos en que, por primera vez, todos los episodios de un lote terminaron en vuelta completa."""
    for row in history:
        if row.get("end_lap") == 1.0:
            return row["steps"]
    return None


def policy_poses(agent, seed: int, circuit: str) -> tuple:
    env = make_env(circuit)
    observation, _ = env.reset(seed=seed)
    poses = []
    while True:
        observation, _, terminated, truncated, _ = env.step(agent.act(observation))
        poses.append((env.state.car.x, env.state.car.y, env.state.car.heading))
        if terminated or truncated:
            return env.track, np.array(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), default="easy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--no-gif", action="store_true")
    args = parse_args(parser, selectors=('circuit',))

    out = results_dir() / "ch02"
    out.mkdir(parents=True, exist_ok=True)
    base = PPOConfig(total_steps=args.steps)
    config = CIRCUITS[args.circuit]()
    env_factory = partial(make_env, args.circuit)
    results = {"metadata": run_metadata(args.seeds), "circuit": args.circuit, "env_config": config,
               "ppo_config": base, "eval_seeds": list(EVAL_SEEDS), "runs": []}

    expert = summarize(evaluate(config, ExpertPolicy(), EVAL_SEEDS))
    results["expert"] = expert
    print(f"experto: vuelta {expert['lap_fraction_mean']:.3f}, {expert['completed_laps']}/{expert['episodes']}"
          f" vueltas, {expert['lap_time_mean']:.1f} s, return {expert['return_mean']:.2f}")

    for seed in args.seeds:
        agent, history = train(env_factory, replace(base, seed=seed), log=lambda row: None)
        save_curves(out / f"ppo_{args.circuit}_seed{seed}.csv", history)
        agent.save(str(out / f"ppo_{args.circuit}_seed{seed}.pt"), replace(base, seed=seed))
        records = evaluate(config, FunctionPolicy(agent.act), EVAL_SEEDS)
        summary = summarize(records)
        run = {"seed": seed, "train_seconds": history[-1]["seconds"],
               "steps_to_all_laps": first_update_with_all_laps(history), "eval": summary,
               "eval_episodes": records}
        results["runs"].append(run)
        lap_time = f"{summary['lap_time_mean']:.1f} s" if summary["lap_time_mean"] else "—"
        print(f"semilla {seed}: {history[-1]['seconds']:.0f} s de entrenamiento, todas vueltas desde "
              f"{run['steps_to_all_laps']} pasos | eval: vuelta {summary['lap_fraction_mean']:.3f}, "
              f"{summary['completed_laps']}/{summary['episodes']} vueltas, {lap_time}, "
              f"freno {summary['mean_brake']:.3f}, fines {summary['end_reasons']}")
        if seed == args.seeds[0] and not args.no_gif:
            track, poses = policy_poses(agent, EVAL_SEEDS[0], args.circuit)
            episode_gif(track, poses[::2], str(results_dir() / f"ch02_ppo_{args.circuit}_lap.gif"), fps=5)

    print(f"guardado en {save_json(f'ch02_ppo_{args.circuit}.json', results)}")


if __name__ == "__main__":
    main()
