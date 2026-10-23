"""Los ejercicios de código del libro: predice y mide.

Cada ejercicio cambia una sola cosa respecto de un experimento de su capítulo y guarda lo que midió en
`results/exercises.json`, con la semilla y el commit. Las soluciones del apéndice B citan ese archivo.

    uv run python experiments/exercises.py --exercise ch02_epsilon
    uv run python experiments/exercises.py            # todos, unos 10 minutos
"""

import argparse
import json
from dataclasses import replace

import numpy as np

from common import parse_args, results_dir, run_metadata, save_json
from ch01_agent_loop import RandomPolicy
from ch02_ppo import EVAL_SEEDS as PPO_EVAL_SEEDS, first_update_with_all_laps
from ch03_bc_dagger import episode_seeds, evaluate_with_shadow
from ch06_rewards import evaluate_policy
from ch07_observations import EVAL_SEEDS as OBS_EVAL_SEEDS, UNSEEN_TRACKS
from algos.bc import BCConfig, record_demonstrations, train_bc
from algos.gail import GAIL, GAILConfig
from algos.ml_minimo import gradient_descent
from algos.ppo import PPOConfig, train
from rlcar import RLCarEnv, presets
from rlcar.evaluate import FunctionPolicy, evaluate, run_episode, summarize
from rlcar.expert import Expert

STEPS = 500_000

# Los dos momentos del experto del capítulo 0: 10° a la izquierda y gira 0,625; 8° a la derecha y −0,5.
TWO_POINTS_X = np.array([[10.0], [-8.0]])
TWO_POINTS_Y = np.array([0.625, -0.5])


def ch00_learning_rate(seed: int) -> dict:
    """Capítulo 0: el descenso de gradiente a cada lado del límite 2/164 ≈ 0,0122."""
    result = {}
    for rate in (0.012, 0.0125):
        _, _, history = gradient_descent(TWO_POINTS_X, TWO_POINTS_Y, rate, steps=200)
        result[str(rate)] = {"loss_after_10": history[9], "loss_after_200": history[-1]}
    return result


class RandomNoBrakePolicy(RandomPolicy):
    """El agente al azar del capítulo 1, pero sin tocar nunca el freno."""

    def act(self, observation, env):
        action = super().act(observation, env)
        action[2] = 0.0
        return action


def ch01_no_brake(seed: int) -> dict:
    """Capítulo 1: el agente al azar sin freno, en el circuito fácil."""
    config = presets.agentic()
    records = [run_episode(config, RandomNoBrakePolicy(s), s)[0] for s in range(seed, seed + 5)]
    return {"seeds": list(range(seed, seed + 5)), **summarize(records), "durations": [r.duration for r in records]}


def ppo_run(make_env, config, seed: int, reward_hook=None):
    return train(make_env, PPOConfig(total_steps=STEPS, seed=seed, **config), log=lambda row: None,
                 reward_hook=reward_hook)


def ch02_epsilon(seed: int) -> dict:
    """Capítulo 2: PPO en el circuito técnico con ε = 0,05 en lugar de 0,2."""
    env_config = presets.technical()
    agent, history = ppo_run(lambda: RLCarEnv(env_config), {"clip_epsilon": 0.05}, seed)
    return {"clip_epsilon": 0.05, "steps_to_all_laps": first_update_with_all_laps(history),
            "train_seconds": history[-1]["seconds"],
            "eval": summarize(evaluate(env_config, FunctionPolicy(agent.act), PPO_EVAL_SEEDS))}


def ch03_three_demos(seed: int) -> dict:
    """Capítulo 3: BC con 3 demostraciones limpias, entre las 2 y las 5 del capítulo."""
    config = presets.technical()
    clean = replace(config, spawn=replace(config.spawn, clean=True))
    runs = []
    for s in range(seed, seed + 5):
        demos = record_demonstrations(clean, Expert(memory=False), episode_seeds(s, "clean", 3))
        policy = train_bc(demos, BCConfig(seed=s))
        result = evaluate_with_shadow(config, policy, memory=False)
        runs.append({"seed": s, "samples": len(demos), "completed_laps": result["completed_laps"],
                     "end_reasons": result["end_reasons"], "disagreement": result["disagreement"]})
    return {"demo_episodes": 3, "runs": runs}


def ch04_no_penalty(seed: int) -> dict:
    """Capítulo 4: GAIL solo, sin la penalización de gradiente."""
    config = presets.technical()
    clean = replace(config, spawn=replace(config.spawn, clean=True))
    env = RLCarEnv(config)
    demos = record_demonstrations(clean, Expert(memory=False), episode_seeds(seed, "clean", 20))
    gail_config = GAILConfig(gail_weight=1.0, env_weight=0.0, gradient_penalty=0.0, seed=seed)
    gail = GAIL(demos, env.action_space.low, env.action_space.high, gail_config)
    agent, history = ppo_run(lambda: RLCarEnv(config), {}, seed, reward_hook=gail)
    d_agent = [row["d_agent"] for row in history]
    return {"gradient_penalty": 0.0, "train_seconds": history[-1]["seconds"],
            "d_expert_last": history[-1]["d_expert"], "d_agent_last": history[-1]["d_agent"],
            "d_agent_min": float(np.min(d_agent)),
            "eval": evaluate_with_shadow(config, agent, memory=False)}


def ch06_target_weight(seed: int) -> dict:
    """Capítulo 6: la velocidad objetivo con peso 0,5 en lugar de 0,25 (el ejercicio 2)."""
    base = presets.technical()
    config = replace(base, reward=replace(base.reward, target_speed_per_sec=0.5))
    agent, history = ppo_run(lambda: RLCarEnv(config), {}, seed)
    return {"target_speed_per_sec": 0.5, "train_seconds": history[-1]["seconds"],
            "eval": evaluate_policy(config, FunctionPolicy(agent.act))}


def ch07_progress_sincos(seed: int) -> dict:
    """Capítulo 7: el progreso como seno y coseno, sin el salto en la meta."""
    base = presets.technical()
    config = replace(base, obs=replace(base.obs, progress_encoding="sincos"))
    agent, history = ppo_run(lambda: RLCarEnv(config), {}, seed)
    policy = FunctionPolicy(agent.act)
    result = {"progress_encoding": "sincos", "train_seconds": history[-1]["seconds"],
              "seen": summarize(evaluate(config, policy, OBS_EVAL_SEEDS))}
    for name, hard in (("unseen", False), ("unseen_hard", True)):
        tracks = (presets.random_tracks(seeds=(UNSEEN_TRACKS[0],), min_corner_radius=None, aggressive=True)
                  if hard else presets.random_tracks(seeds=(UNSEEN_TRACKS[0],)))
        records = []
        for track_seed, eval_seed in zip(UNSEEN_TRACKS, OBS_EVAL_SEEDS):
            track_config = replace(tracks, track=replace(tracks.track, random_seeds=(track_seed,)), obs=config.obs)
            records.append(run_episode(track_config, policy, eval_seed)[0])
        result[name] = summarize(records)
    return result


# Los ejercicios que corren varias semillas a partir de `--seed`; los demás usan solo esa.
SEED_COUNT = {"ch01_no_brake": 5, "ch03_three_demos": 5}

EXERCISES = {f.__name__: f for f in (ch00_learning_rate, ch01_no_brake, ch02_epsilon, ch03_three_demos,
                                     ch04_no_penalty, ch06_target_weight, ch07_progress_sincos)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exercise", nargs="+", choices=list(EXERCISES), default=list(EXERCISES))
    parser.add_argument("--seed", type=int, default=0)
    args = parse_args(parser, selectors=('exercise',))

    path = results_dir() / "exercises.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.exercise:
        seeds = range(args.seed, args.seed + SEED_COUNT.get(name, 1))
        result = {"metadata": run_metadata(seeds), **EXERCISES[name](args.seed)}
        results[name] = result
        save_json("exercises.json", dict(sorted(results.items())))
        print(f"{name}: {json.dumps({k: v for k, v in result.items() if k != 'metadata'}, default=str)[:400]}",
              flush=True)
    print(f"guardado en {path}")


if __name__ == "__main__":
    main()
