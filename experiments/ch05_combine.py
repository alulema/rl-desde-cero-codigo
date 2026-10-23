"""Capítulo 5: las variantes A–E del capítulo 5, y si la imitación conserva la respuesta a la directiva.

En el circuito técnico, con las mismas 20 demostraciones limpias del experto sin memoria para todas las
variantes que las usan, 5 semillas y 500 000 pasos por corrida de PPO:

- A. PPO desde cero, solo la recompensa del entorno (y `A_long`, con 1 000 000 de pasos, el presupuesto
  de E).
- B. BC puro.
- C. GAIL 0.15 + entorno.
- D. Simultánea: BC (0.5, apagado al 20 %) + GAIL 0.15 + entorno, en una sola corrida.
- E. Secuencial: D, y después 500 000 pasos más solo con la recompensa del entorno, arrancando de D con un
  optimizador nuevo y los schedules desde cero (como `--initialize-from`).

La recompensa no lee la directiva. Cada variante se evalúa con la agresión al azar (arranques con ruido,
experto en la sombra, como en el capítulo 3) y con la agresión forzada a 0.15 (conservar) y 0.85
(atacar): la diferencia de tiempo por vuelta entre las dos mide cuánto responde a la directiva.

    uv run python experiments/ch05_combine.py --seeds 0 1 2 3 4
"""

import argparse
from dataclasses import replace

from common import parse_args, results_dir, run_metadata, save_json
from ch02_ppo import save_curves
from ch03_bc_dagger import describe, episode_seeds, evaluate_with_shadow
from algos.bc import BCConfig, record_demonstrations, train_bc
from algos.combine import BCAuxConfig, BCAuxLoss
from algos.gail import GAIL, GAILConfig
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.expert import Expert

DEMO_EPISODES = 20
DIRECTIVES = {"conserve": 0.15, "attack": 0.85}
VARIANTS = ["A", "A_long", "B", "C", "D", "E"]


def make_env():
    return RLCarEnv(presets.technical())


def evaluate_variant(config, policy) -> dict:
    result = {"random_directive": evaluate_with_shadow(config, policy, memory=False)}
    for name, aggression in DIRECTIVES.items():
        result[name] = evaluate_with_shadow(replace(config, aggression=aggression), policy, memory=False)
    conserve, attack = result["conserve"]["lap_time_mean"], result["attack"]["lap_time_mean"]
    result["conserve_minus_attack_s"] = conserve - attack if conserve and attack else None
    return result


def report(seed, name, result, seconds=None):
    gap = result["conserve_minus_attack_s"]
    gap = f"{gap:+.1f} s" if gap is not None else "—"
    took = f" ({seconds:.0f} s)" if seconds is not None else ""
    print(f"semilla {seed} {name}{took}: " + describe("azar", result["random_directive"])
          + f" | conservar − atacar {gap}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parse_args(parser)

    out = results_dir() / "ch05"
    out.mkdir(parents=True, exist_ok=True)
    config = presets.technical()
    clean = replace(config, spawn=replace(config.spawn, clean=True))
    env = make_env()
    ppo = PPOConfig(total_steps=args.steps)
    results = {"metadata": run_metadata(args.seeds), "env_config": config, "ppo_config": ppo,
               "gail_config": GAILConfig(), "bc_aux_config": BCAuxConfig(), "bc_config": BCConfig(),
               "demo_episodes": DEMO_EPISODES, "directives": DIRECTIVES,
               "expert": evaluate_variant(config, None), "runs": []}
    report("-", "experto", results["expert"])

    for seed in args.seeds:
        demos = record_demonstrations(clean, Expert(memory=False), episode_seeds(seed, "clean", DEMO_EPISODES))
        run = {"seed": seed, "demo_samples": len(demos), "variants": {}}

        def ppo_variant(name, steps, **kwargs):
            agent, history = train(make_env, replace(ppo, total_steps=steps, seed=seed), log=lambda row: None,
                                   **kwargs)
            save_curves(out / f"{name}_seed{seed}.csv", history)
            result = {"steps": steps, "train_seconds": history[-1]["seconds"], **evaluate_variant(config, agent)}
            run["variants"][name] = result
            report(seed, name, result, history[-1]["seconds"])
            return agent

        def gail():
            return GAIL(demos, env.action_space.low, env.action_space.high, GAILConfig(seed=seed))

        ppo_variant("A", args.steps)
        ppo_variant("A_long", 2 * args.steps)

        bc = train_bc(demos, BCConfig(seed=seed))
        run["variants"]["B"] = {"steps": 0, **evaluate_variant(config, bc)}
        report(seed, "B", run["variants"]["B"])

        ppo_variant("C", args.steps, reward_hook=gail())
        d = ppo_variant("D", args.steps, reward_hook=gail(),
                        update_hook=BCAuxLoss(demos, ppo.learning_rate, BCAuxConfig(seed=seed)))
        ppo_variant("E", args.steps, init=d)
        run["variants"]["E"]["steps"] = 2 * args.steps  # D + el fine-tuning
        results["runs"].append(run)
        save_json("ch05_combine.json", results)  # después de cada semilla, por si se corta

    print(f"guardado en {save_json('ch05_combine.json', results)}")


if __name__ == "__main__":
    main()
