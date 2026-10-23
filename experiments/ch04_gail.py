"""Capítulo 4: GAIL solo, y GAIL + la recompensa del entorno.

En el circuito técnico, con las demostraciones del experto sin memoria (20 episodios desde arranques
limpios, como el grabador de Agentic Racing). Dos variantes:

- `gail_only`: solo la recompensa de GAIL (peso 1.0). Las terminaciones del entorno siguen cortando el
  episodio, pero sin castigo ni bono.
- `gail_env`: la recompensa del entorno (1.0) más GAIL (0.15), la configuración preparada para Agentic
  Racing.

La referencia sin imitación es PPO en el mismo circuito (`results/ch02_ppo_technical.json`). La evaluación
es la del capítulo 3: arranques con ruido, con el experto en la sombra.

    uv run python experiments/ch04_gail.py --variants gail_only gail_env --seeds 0 1 2 3 4
"""

import argparse
from dataclasses import replace
from functools import partial

from common import parse_args, results_dir, run_metadata, save_json
from ch02_ppo import save_curves
from ch03_bc_dagger import describe, episode_seeds, evaluate_with_shadow
from algos.bc import record_demonstrations
from algos.gail import GAIL, GAILConfig
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.expert import Expert

VARIANTS = {
    "gail_only": GAILConfig(gail_weight=1.0, env_weight=0.0),
    "gail_env": GAILConfig(gail_weight=0.15, env_weight=1.0),
}
DEMO_EPISODES = 20


def make_env():
    return RLCarEnv(presets.technical())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser, selectors=('variants',))

    out = results_dir() / "ch04"
    out.mkdir(parents=True, exist_ok=True)
    config = presets.technical()
    clean = replace(config, spawn=replace(config.spawn, clean=True))
    env = make_env()
    for variant in args.variants:
        results = {"metadata": run_metadata(args.seeds), "variant": variant, "env_config": config,
                   "gail_config": VARIANTS[variant], "ppo_config": PPOConfig(total_steps=args.steps),
                   "demo_episodes": DEMO_EPISODES, "runs": []}
        for seed in args.seeds:
            demos = record_demonstrations(clean, Expert(memory=False), episode_seeds(seed, "clean", DEMO_EPISODES))
            gail = GAIL(demos, env.action_space.low, env.action_space.high, replace(VARIANTS[variant], seed=seed))
            agent, history = train(make_env, PPOConfig(total_steps=args.steps, seed=seed), log=lambda row: None,
                                   reward_hook=gail)
            save_curves(out / f"{variant}_seed{seed}.csv", history)
            result = {"seed": seed, "train_seconds": history[-1]["seconds"], "demo_samples": len(demos),
                      "last_batch": history[-1], "eval": evaluate_with_shadow(config, agent, memory=False)}
            results["runs"].append(result)
            last = history[-1]
            print(f"[{variant}] semilla {seed} ({last['seconds']:.0f} s, D experto {last['d_expert']:.2f}, "
                  f"D agente {last['d_agent']:.2f}) " + describe("eval", result["eval"]), flush=True)
        print(f"guardado en {save_json(f'ch04_gail_{variant}.json', results)}")


if __name__ == "__main__":
    main()
