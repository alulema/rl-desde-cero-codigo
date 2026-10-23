"""Capítulo 3: behavioral cloning desde arranques limpios y con ruido, y después DAgger.

En el circuito técnico (`presets.technical()`). Para cada semilla y cada cantidad de episodios de
demostración:

- BC limpio: el experto graba desde arranques limpios (centrado, alineado, a 14 m/s), como el grabador
  de Agentic Racing.
- BC con ruido: el experto graba desde los mismos arranques con ruido que se usan para evaluar.
- DAgger: parte de BC limpio con cada cantidad de `--dagger-from` episodios y suma, en cada iteración, episodios manejados
  por la policy y etiquetados por el experto.

Todo se evalúa desde arranques con ruido (semillas 1000–1019), con el experto en la sombra: además de la
fracción de vuelta y los motivos de fin, se mide cuánto difiere la policy del experto en los estados que
ella misma visita (`disagreement`) y cómo cambia eso a lo largo del episodio.

    uv run python experiments/ch03_bc_dagger.py --expert memoryless --seeds 0 1 2 3 4
"""

import argparse
from collections import Counter
from dataclasses import replace
from functools import partial

import numpy as np

from common import parse_args, run_metadata, save_json
from algos.bc import BCConfig, record_demonstrations, rollout, train_bc
from algos.dagger import DAggerConfig, dagger
from rlcar import presets
from rlcar.expert import Expert

EVAL_SEEDS = range(1000, 1020)
TIME_BINS = (0, 5, 10, 20, 30, 60)  # segundos, para ver si el desacuerdo crece durante el episodio


def episode_seeds(seed: int, kind: str, n: int, iteration: int = 0) -> list[int]:
    """Semillas de los episodios de entrenamiento: distintas por semilla de experimento, por tipo y por
    iteración de DAgger, y nunca iguales a las de evaluación."""
    offset = {"clean": 0, "noisy": 1_000, "dagger": 2_000 + 100 * iteration}[kind]
    return [100_000 + 10_000 * seed + offset + k for k in range(n)]


def evaluate_with_shadow(config, policy, memory: bool) -> dict:
    """Evalúa desde arranques con ruido. Con `policy=None` maneja el experto (desacuerdo 0)."""
    driver = policy.act if policy is not None else None
    rollouts = [rollout(config, Expert(memory=memory), s, driver=driver) for s in EVAL_SEEDS]
    decision_dt = config.physics_dt * config.decision_period
    by_bin = []
    for start, end in zip(TIME_BINS[:-1], TIME_BINS[1:]):
        values = [d for r in rollouts for k, d in enumerate(r.disagreement) if start <= k * decision_dt < end]
        by_bin.append({"from_s": start, "to_s": end, "steps": len(values),
                       "disagreement": float(np.mean(values)) if values else None})
    laps = [r.duration for r in rollouts if r.end_reason == "lap"]
    driven = np.concatenate([r.driven_actions for r in rollouts])
    return {
        "episodes": len(rollouts),
        "lap_fraction_mean": float(np.mean([r.lap_fraction for r in rollouts])),
        "lap_fraction_min": float(np.min([r.lap_fraction for r in rollouts])),
        "completed_laps": len(laps),
        "lap_time_mean": float(np.mean(laps)) if laps else None,
        "end_reasons": dict(Counter(r.end_reason for r in rollouts)),
        "mean_brake": float(driven[:, 2].mean()),
        "brake_over_threshold": float((driven[:, 2] > 0.01).mean()),  # el freno anula el acelerador
        "disagreement": float(np.mean(np.concatenate([r.disagreement for r in rollouts]))),
        "disagreement_by_time": by_bin,
        "mean_abs_lateral": float(np.mean(np.abs(np.concatenate([r.lateral for r in rollouts])))),
    }


def describe(name: str, result: dict) -> str:
    lap_time = f"{result['lap_time_mean']:.1f} s" if result["lap_time_mean"] else "—"
    return (f"{name}: vuelta {result['lap_fraction_mean']:.2f}, {result['completed_laps']}/{result['episodes']}, "
            f"{lap_time}, desacuerdo {result['disagreement']:.3f}, freno>0.01 {result['brake_over_threshold']:.2f}, "
            f"{result['end_reasons']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expert", choices=["memoryless", "memory"], default="memoryless")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--demos", type=int, nargs="+", default=[1, 2, 5, 10, 20, 35])
    parser.add_argument("--dagger-from", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--dagger-iterations", type=int, default=6)
    parser.add_argument("--dagger-episodes", type=int, default=5)
    args = parse_args(parser, selectors=('expert',))

    memory = args.expert == "memory"
    config = presets.technical()
    clean = replace(config, spawn=replace(config.spawn, clean=True))
    dagger_config = DAggerConfig(iterations=args.dagger_iterations, episodes_per_iteration=args.dagger_episodes)
    results = {"metadata": run_metadata(args.seeds), "expert": args.expert, "env_config": config,
               "bc_config": BCConfig(), "dagger_config": dagger_config, "eval_seeds": list(EVAL_SEEDS),
               "expert_eval": evaluate_with_shadow(config, None, memory), "runs": []}
    print(f"[{args.expert}] " + describe("experto", results["expert_eval"]), flush=True)

    for seed in args.seeds:
        bc_config = BCConfig(seed=seed)
        run = {"seed": seed, "bc": [], "dagger": []}
        for kind, spawn_config in (("clean", clean), ("noisy", config)):
            for n in args.demos:
                demos = record_demonstrations(spawn_config, Expert(memory=memory), episode_seeds(seed, kind, n))
                policy = train_bc(demos, bc_config)
                result = {"spawn": kind, "demo_episodes": n, "samples": len(demos),
                          "train_loss": policy.loss(demos), **evaluate_with_shadow(config, policy, memory)}
                run["bc"].append(result)
                print(f"[{args.expert}] semilla {seed} " + describe(f"BC {kind} ×{n} ({len(demos)} pares)", result),
                      flush=True)

        def on_iteration(start_episodes, i, policy, rollouts, data):
            result = {"start_episodes": start_episodes, "iteration": i, "samples": len(data), "train_loss": policy.loss(data),
                      "rollout_lap_fraction": [r.lap_fraction for r in rollouts],
                      **evaluate_with_shadow(config, policy, memory)}
            run["dagger"].append(result)
            print(f"[{args.expert}] semilla {seed} " + describe(f"DAgger ×{start_episodes} it {i} ({len(data)} pares)", result),
                  flush=True)

        for n in args.dagger_from:
            start = record_demonstrations(clean, Expert(memory=memory), episode_seeds(seed, "clean", n))
            dagger(config, Expert(memory=memory), start, replace(dagger_config, bc=bc_config),
                   lambda i: episode_seeds(seed, "dagger", args.dagger_episodes, iteration=i),
                   partial(on_iteration, n))
        results["runs"].append(run)

    print(f"guardado en {save_json(f'ch03_bc_dagger_{args.expert}.json', results)}")


if __name__ == "__main__":
    main()
