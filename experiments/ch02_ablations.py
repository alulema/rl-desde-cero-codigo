"""Capítulo 2: ¿qué cambia si `rlcar` se parece más a Agentic Racing?

PPO aprende `rlcar` en segundos, y en Agentic Racing no aprendió en millones de pasos. Cada variante cambia
una sola cosa respecto al circuito técnico (o al fácil), para ver cuál importa:

- `agentic_grip`: el agarre de Agentic Racing, que crea velocidad al girar.
- `grip_bug`: el agarre de antes del arreglo, que frena el auto en cada giro.
- `agentic_size`: el circuito fijo de Agentic Racing a tamaño real (~2 km, esquinas de 120 m, 120 s).
- `random_tracks`: 8 pistas aleatorias, una al azar por episodio (race01–08 corrieron en pistas
  procedurales, no en el circuito fijo).

    uv run python experiments/ch02_ablations.py --variants agentic_grip grip_bug --seeds 0 1 2
"""

import argparse
from dataclasses import replace
from functools import partial

from common import parse_args, results_dir, run_metadata, save_json
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.env import TrackConfig
from rlcar.evaluate import ExpertPolicy, FunctionPolicy, evaluate, summarize
from rlcar.terminations import TerminationConfig

EVAL_SEEDS = range(1000, 1010)


def agentic_size():
    base = presets.agentic()
    return replace(base, track=TrackConfig(half_length=350, half_height=200, corner_radius=120),
                   termination=replace(base.termination, time_limit_seconds=120.0))


VARIANTS = {
    "agentic_grip": lambda: presets.agentic_grip(presets.technical()),
    "grip_bug": lambda: presets.grip_bug(presets.technical()),
    "agentic_size": agentic_size,
    "random_tracks": lambda: presets.random_tracks(seeds=tuple(range(8))),
}


def make_env(variant: str):
    return RLCarEnv(VARIANTS[variant]())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser, selectors=('variants',))
    (results_dir() / "ch02").mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        config = VARIANTS[variant]()
        results = {"metadata": run_metadata(args.seeds), "variant": variant, "env_config": config,
                   "ppo_config": PPOConfig(total_steps=args.steps), "eval_seeds": list(EVAL_SEEDS), "runs": []}
        results["expert"] = summarize(evaluate(config, ExpertPolicy(), EVAL_SEEDS))
        e = results["expert"]
        print(f"[{variant}] experto: {e['completed_laps']}/{e['episodes']} vueltas, vuelta {e['lap_fraction_mean']:.2f}, "
              f"{e['lap_time_mean'] or float('nan'):.1f} s", flush=True)
        for seed in args.seeds:
            agent, history = train(partial(make_env, variant), PPOConfig(total_steps=args.steps, seed=seed),
                                   log=lambda row: None)
            summary = summarize(evaluate(config, FunctionPolicy(agent.act), EVAL_SEEDS))
            last = history[-1]
            results["runs"].append({"seed": seed, "train_seconds": last["seconds"], "last_batch": last,
                                    "eval": summary})
            lap_time = f"{summary['lap_time_mean']:.1f} s" if summary["lap_time_mean"] else "—"
            print(f"[{variant}] semilla {seed}: {last['seconds']:.0f} s | último lote: vuelta "
                  f"{last.get('lap_fraction', float('nan')):.2f} | eval: vuelta {summary['lap_fraction_mean']:.2f} "
                  f"(mín {summary['lap_fraction_min']:.2f}), {summary['completed_laps']}/{summary['episodes']}, "
                  f"{lap_time}, freno {summary['mean_brake']:.3f}, {summary['end_reasons']}", flush=True)
        save_json(f"ch02_ablation_{variant}.json", results)


if __name__ == "__main__":
    main()
