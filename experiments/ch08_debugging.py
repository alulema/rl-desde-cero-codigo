"""Capítulo 8: un entorno con bugs plantados, y la secuencia de depuración aplicada.

Cuatro bugs, cada uno de un tipo que tuvo Agentic Racing:

- `grip_bug`: el agarre borra la velocidad lateral en lugar de girarla (cada giro frena el auto).
- `zero_observation`: la observación llega con el tamaño correcto, pero solo los 9 rayos tienen datos; el
  resto, ceros.
- `short_time_limit`: el límite de tiempo (25 s) es más corto que una vuelta del experto (≈ 30 s).
- `baseline_sign`: la línea base (el experto) corrige hacia el centro con el signo invertido. El entorno
  está sano: el bug está en la referencia contra la que se compara.

Y dos escenarios de control: `clean` (sin bugs) y `all_bugs` (los cuatro juntos: "el agente no aprende").

Para cada escenario, en el circuito técnico:
1. Los chequeos de `rlcar.diagnostics`, que no entrenan nada: observación, efecto de la acción,
   recompensa en casos conocidos, terminaciones y línea base.
2. PPO con 5 semillas y 500 000 pasos, evaluado con el evaluador independiente (arranques con ruido,
   semillas 1000–1019). Si el escenario cambia el límite de tiempo, se evalúa también con el de 60 s,
   para separar lo que el límite le hizo al aprendizaje de lo que le hizo a la medición. `baseline_sign`
   no entrena: su entorno es el de `clean`.

    uv run python experiments/ch08_debugging.py --seeds 0 1 2 3 4
"""

import argparse
from dataclasses import replace
from functools import partial

from common import parse_args, results_dir, run_metadata, save_json
from ch02_ppo import save_curves
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.diagnostics import run_all
from rlcar.evaluate import ExpertPolicy, FunctionPolicy, evaluate, summarize

EVAL_SEEDS = range(1000, 1020)
SCENARIOS = {  # nombre: (entorno, la línea base tiene el signo invertido)
    "clean": (lambda: presets.technical(), False),
    "grip_bug": (lambda: presets.grip_bug(presets.technical()), False),
    "zero_observation": (lambda: presets.zero_filled_observation(presets.technical()), False),
    "short_time_limit": (lambda: presets.short_time_limit(presets.technical()), False),
    "baseline_sign": (lambda: presets.technical(), True),
    "all_bugs": (lambda: presets.all_bugs(presets.technical()), True),
}
TRAIN = ["clean", "grip_bug", "zero_observation", "short_time_limit", "all_bugs"]


def make_env(scenario: str):
    return RLCarEnv(SCENARIOS[scenario][0]())


def describe(result: dict) -> str:
    lap_time = f"{result['lap_time_mean']:.1f} s" if result["lap_time_mean"] else "—"
    return (f"{result['completed_laps']}/{result['episodes']} vueltas, fracción {result['lap_fraction_mean']:.2f}, "
            f"{lap_time}, return {result['return_mean']:.2f}, {result['end_reasons']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser)

    out = results_dir() / "ch08"
    out.mkdir(parents=True, exist_ok=True)
    clean_limit = presets.technical().termination.time_limit_seconds
    results = {"metadata": run_metadata(args.seeds), "ppo_config": PPOConfig(total_steps=args.steps),
               "eval_seeds": list(EVAL_SEEDS), "scenarios": {}}

    for name in args.scenarios:
        config = SCENARIOS[name][0]()
        baseline = ExpertPolicy(memory=False, sign_bug=SCENARIOS[name][1])
        checks = run_all(config, baseline, EVAL_SEEDS)
        entry = {"env_config": config, "baseline_sign_bug": SCENARIOS[name][1], "checks": checks, "runs": []}
        print(f"== {name}", flush=True)
        for check, result in checks.items():
            print(f"   {'MIRAR' if result['flag'] else 'ok   '} {check}: {result['finding']}", flush=True)

        if name in TRAIN:
            fixed_limit = replace(config, termination=replace(config.termination, time_limit_seconds=clean_limit))
            for seed in args.seeds:
                agent, history = train(partial(make_env, name), PPOConfig(total_steps=args.steps, seed=seed),
                                       log=lambda row: None)
                save_curves(out / f"{name}_seed{seed}.csv", history)
                policy = FunctionPolicy(agent.act)
                run = {"seed": seed, "train_seconds": history[-1]["seconds"],
                       "evaluation": summarize(evaluate(config, policy, EVAL_SEEDS))}
                line = f"[{name}] semilla {seed} ({history[-1]['seconds']:.0f} s): " + describe(run["evaluation"])
                if config.termination.time_limit_seconds != clean_limit:
                    run["evaluation_60s"] = summarize(evaluate(fixed_limit, policy, EVAL_SEEDS))
                    line += f" | con {clean_limit:.0f} s: " + describe(run["evaluation_60s"])
                entry["runs"].append(run)
                print(line, flush=True)
        results["scenarios"][name] = entry
        save_json("ch08_debugging.json", results)

    print(f"guardado en {save_json('ch08_debugging.json', results)}")


if __name__ == "__main__":
    main()
