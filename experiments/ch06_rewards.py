"""Capítulo 6: ablaciones de recompensa, y el atajo de la cuarta corrida.

En el circuito técnico, PPO con 5 semillas y 500 000 pasos por cada recompensa:

- `agentic`: la de race07/08, la referencia.
- `sparse`: solo el bono por vuelta.
- `no_target_speed`: sin el término de velocidad objetivo ni el castigo por ir lento.
- `with_directive`: la velocidad objetivo se escala con la agresión (la propuesta del capítulo 6).
- `race04`: la recompensa y las terminaciones con las que apareció el atajo.
- `survival_bias`: un premio constante por seguir vivo, sin bono por vuelta.
- `race04_aggressive`: `race04` en 8 pistas aleatorias con curvas muy cerradas, como las de race01–07, en
  lugar del circuito técnico. Es la condición en que apareció el atajo: la vuelta fuera de alcance.

Cada policy se evalúa con el evaluador independiente de la recompensa (arranques con ruido, semillas
1000–1019), con la agresión al azar y forzada a 0.15 y 0.85. Se reporta también cuánto sumó cada término
de la recompensa en el episodio, porque el return solo no dice qué se está pagando.

Antes de entrenar, la contabilidad del atajo: el return de una vuelta del experto frente al de "rodar
245 m y detenerse" (el comportamiento de race04), con cada recompensa.

    uv run python experiments/ch06_rewards.py --seeds 0 1 2 3 4
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from functools import partial

import numpy as np

from common import parse_args, results_dir, run_metadata, save_json
from ch02_ppo import save_curves
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.evaluate import FunctionPolicy, evaluate, summarize
from rlcar.expert import Expert

EVAL_SEEDS = range(1000, 1020)
DIRECTIVES = {"conserve": 0.15, "attack": 0.85}
SHORTCUT_METRES = 245.0  # el avance del atajo de race04, según la bitácora de Agentic Racing


def reward_config(name: str):
    if name == "race04_aggressive":
        # Las pistas de race01–07 antes de suavizarse: curvas más cerradas que el radio de giro del auto.
        return presets.random_tracks(presets.race04(), min_corner_radius=None, aggressive=True)
    return presets.technical(getattr(presets, name)())


VARIANTS = ["agentic", "sparse", "no_target_speed", "with_directive", "race04", "survival_bias",
            "race04_aggressive"]


class ShortcutPolicy:
    """Maneja como el experto hasta los 245 m y después frena hasta detenerse."""

    def __init__(self):
        self.expert = Expert(memory=False)

    def reset(self):
        self.expert.reset()

    def act(self, observation, env):
        if env.state.lap_metres < SHORTCUT_METRES:
            return self.expert.act(env.state)
        return np.array([0.0, 0.0, 1.0])


class ExpertDriver(ShortcutPolicy):
    def act(self, observation, env):
        return self.expert.act(env.state)


def term_means(records) -> dict:
    totals = defaultdict(float)
    for r in records:
        for term, value in r.reward_terms.items():
            totals[term] += value
    return {term: value / len(records) for term, value in sorted(totals.items())}


def evaluate_policy(config, policy) -> dict:
    records = evaluate(config, policy, EVAL_SEEDS)
    result = {"random_directive": {**summarize(records), "reward_terms": term_means(records)}}
    for name, aggression in DIRECTIVES.items():
        result[name] = summarize(evaluate(replace(config, aggression=aggression), policy, EVAL_SEEDS))
    conserve, attack = result["conserve"]["lap_time_mean"], result["attack"]["lap_time_mean"]
    result["conserve_minus_attack_s"] = conserve - attack if conserve and attack else None
    return result


def describe(result: dict) -> str:
    r = result["random_directive"]
    lap_time = f"{r['lap_time_mean']:.1f} s" if r["lap_time_mean"] else "—"
    gap = result["conserve_minus_attack_s"]
    gap = f"{gap:+.1f} s" if gap is not None else "—"
    terms = ", ".join(f"{k} {v:+.2f}" for k, v in r["reward_terms"].items())
    return (f"{r['completed_laps']}/{r['episodes']} vueltas, {lap_time}, return {r['return_mean']:.2f}, "
            f"{r['end_reasons']}, conservar − atacar {gap} | {terms}")


def shortcut_accounting() -> dict:
    """El return de la vuelta del experto y el del atajo, con cada recompensa (arranques limpios)."""
    table = {}
    for name in VARIANTS:
        config = reward_config(name)
        clean = replace(config, spawn=replace(config.spawn, clean=True))
        table[name] = {}
        for label, policy in (("expert_lap", ExpertDriver()), ("shortcut", ShortcutPolicy())):
            records = evaluate(clean, policy, EVAL_SEEDS)
            table[name][label] = {"return_mean": float(np.mean([r.episode_return for r in records])),
                                  "discounted_return_mean": float(np.mean([r.discounted_return
                                                                           for r in records])),
                                  "end_reasons": summarize(records)["end_reasons"],
                                  "reward_terms": term_means(records)}
        lap, cut = table[name]["expert_lap"], table[name]["shortcut"]
        print(f"[{name}] vuelta del experto: return {lap['return_mean']:.2f} {lap['end_reasons']} | atajo: "
              f"return {cut['return_mean']:.2f} {cut['end_reasons']}", flush=True)
    return table


def make_env(name: str):
    return RLCarEnv(reward_config(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser)

    out = results_dir() / "ch06"
    out.mkdir(parents=True, exist_ok=True)
    results = {"metadata": run_metadata(args.seeds), "ppo_config": PPOConfig(total_steps=args.steps),
               "eval_seeds": list(EVAL_SEEDS), "directives": DIRECTIVES,
               "shortcut_accounting": shortcut_accounting(), "variants": {}}
    for name in args.variants:
        config = reward_config(name)
        entry = {"env_config": config, "runs": []}
        for seed in args.seeds:
            agent, history = train(partial(make_env, name), PPOConfig(total_steps=args.steps, seed=seed),
                                   log=lambda row: None)
            save_curves(out / f"{name}_seed{seed}.csv", history)
            result = {"seed": seed, "train_seconds": history[-1]["seconds"],
                      **evaluate_policy(config, FunctionPolicy(agent.act))}
            entry["runs"].append(result)
            print(f"[{name}] semilla {seed} ({history[-1]['seconds']:.0f} s): " + describe(result), flush=True)
        results["variants"][name] = entry
        save_json("ch06_rewards.json", results)

    print(f"guardado en {save_json('ch06_rewards.json', results)}")


if __name__ == "__main__":
    main()
