"""Capítulo 7: ablaciones de observación, y una prueba en pistas no vistas.

PPO con la recompensa de referencia (`presets.agentic()`), 5 semillas y 500 000 pasos, con cinco
observaciones:

- `full`: las 18 de siempre.
- `no_progress`: sin el progreso en la vuelta (0–1), el número con el que se puede memorizar un circuito.
- `no_curvature`: sin las 3 ventanas de curvatura por delante; solo los rayos anticipan la curva.
- `previous_steer`: con el volante del paso anterior.
- `stacked`: las últimas 4 observaciones apiladas.

Y dos formas de entrenar:

- `technical`: siempre el circuito técnico.
- `random8`: una de 8 pistas aleatorias suaves por episodio (semillas 0–7).

Cada policy se evalúa en su distribución de entrenamiento (arranques con ruido, semillas 1000–1019), en
20 pistas aleatorias suaves que nunca vio (semillas de pista 100–119, un episodio en cada una) y, como
prueba más exigente, en 20 pistas no vistas con curvas muy cerradas (`random_tracks(aggressive=True)` sin
radio mínimo, las de race01–07).

    uv run python experiments/ch07_observations.py --seeds 0 1 2 3 4
"""

import argparse
from dataclasses import replace
from functools import partial

from common import parse_args, results_dir, run_metadata, save_json
from ch02_ppo import save_curves
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.evaluate import ExpertPolicy, FunctionPolicy, evaluate, run_episode, summarize
from rlcar.observations import ObsConfig

EVAL_SEEDS = range(1000, 1020)
UNSEEN_TRACKS = range(100, 120)
OBSERVATIONS = {
    "full": ObsConfig(),
    "no_progress": ObsConfig(progress=False),
    "no_curvature": ObsConfig(curvature_windows=()),
    "previous_steer": ObsConfig(previous_steer=True),
    "stacked": ObsConfig(stack=4),
}
TRAINING = {
    "technical": lambda: presets.technical(),
    "random8": lambda: presets.random_tracks(seeds=tuple(range(8))),
}


def env_config(training: str, observation: str):
    return replace(TRAINING[training](), obs=OBSERVATIONS[observation])


def make_env(training: str, observation: str):
    return RLCarEnv(env_config(training, observation))


def evaluate_unseen(observation: str, policy, hard: bool = False) -> dict:
    """Un episodio en cada pista no vista, con la misma recompensa y las mismas terminaciones."""
    if hard:
        base = presets.random_tracks(seeds=(UNSEEN_TRACKS[0],), min_corner_radius=None, aggressive=True)
    else:
        base = presets.random_tracks(seeds=(UNSEEN_TRACKS[0],))
    records = []
    for track_seed, seed in zip(UNSEEN_TRACKS, EVAL_SEEDS):
        config = replace(base, track=replace(base.track, random_seeds=(track_seed,)), obs=OBSERVATIONS[observation])
        records.append(run_episode(config, policy, seed)[0])
    return summarize(records)


def describe(name: str, result: dict) -> str:
    lap_time = f"{result['lap_time_mean']:.1f} s" if result["lap_time_mean"] else "—"
    return (f"{name}: {result['completed_laps']}/{result['episodes']} vueltas, fracción "
            f"{result['lap_fraction_mean']:.2f}, {lap_time}, {result['end_reasons']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", nargs="+", choices=sorted(TRAINING), default=sorted(TRAINING))
    parser.add_argument("--observations", nargs="+", choices=list(OBSERVATIONS), default=list(OBSERVATIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser)

    out = results_dir() / "ch07"
    out.mkdir(parents=True, exist_ok=True)
    results = {"metadata": run_metadata(args.seeds), "ppo_config": PPOConfig(total_steps=args.steps),
               "eval_seeds": list(EVAL_SEEDS), "unseen_tracks": list(UNSEEN_TRACKS),
               "observations": OBSERVATIONS, "experiments": {}}
    expert = {"technical": summarize(evaluate(env_config("technical", "full"), ExpertPolicy(memory=False), EVAL_SEEDS)),
              "random8": summarize(evaluate(env_config("random8", "full"), ExpertPolicy(memory=False), EVAL_SEEDS)),
              "unseen": evaluate_unseen("full", ExpertPolicy(memory=False)),
              "unseen_hard": evaluate_unseen("full", ExpertPolicy(memory=False), hard=True)}
    results["expert"] = expert
    for name, result in expert.items():
        print(describe(f"experto, {name}", result), flush=True)

    for training in args.training:
        for observation in args.observations:
            key = f"{training}/{observation}"
            entry = {"env_config": env_config(training, observation), "runs": []}
            for seed in args.seeds:
                agent, history = train(partial(make_env, training, observation),
                                       PPOConfig(total_steps=args.steps, seed=seed), log=lambda row: None)
                save_curves(out / f"{training}_{observation}_seed{seed}.csv", history)
                policy = FunctionPolicy(agent.act)
                run = {"seed": seed, "train_seconds": history[-1]["seconds"],
                       "seen": summarize(evaluate(env_config(training, observation), policy, EVAL_SEEDS)),
                       "unseen": evaluate_unseen(observation, policy),
                       "unseen_hard": evaluate_unseen(observation, policy, hard=True)}
                entry["runs"].append(run)
                print(f"[{key}] semilla {seed} ({history[-1]['seconds']:.0f} s) " + describe("vistas", run["seen"])
                      + " | " + describe("no vistas", run["unseen"])
                      + " | " + describe("no vistas difíciles", run["unseen_hard"]), flush=True)
            results["experiments"][key] = entry
            save_json("ch07_observations.json", results)

    print(f"guardado en {save_json('ch07_observations.json', results)}")


if __name__ == "__main__":
    main()
