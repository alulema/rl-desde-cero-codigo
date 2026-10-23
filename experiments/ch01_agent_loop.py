"""Capítulo 1: el ciclo agente–entorno, con un agente al azar y con el experto.

Corre episodios en el circuito fijo con la recompensa de Agentic Racing (`presets.agentic()`) y guarda,
por episodio, cómo terminó y cuánto aportó cada término de la recompensa. Además guarda un GIF de una
vuelta del experto.

    uv run python experiments/ch01_agent_loop.py --seeds 0 1 2 3 4
"""

import argparse

import numpy as np

from common import parse_args, results_dir, run_metadata, save_json
from rlcar import RLCarEnv, presets
from rlcar.evaluate import ExpertPolicy, run_episode, summarize
from rlcar.expert import Expert
from rlcar.render import episode_gif


class RandomPolicy:
    """Acciones uniformes en el espacio de acciones, con su propia semilla."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, observation, env):
        return self.rng.uniform(env.action_space.low, env.action_space.high)


def expert_lap_poses(seed: int) -> tuple:
    """Las poses (x, y, rumbo) de una vuelta del experto, decisión por decisión."""
    env = RLCarEnv(presets.agentic())
    env.reset(seed=seed)
    expert, poses = Expert(), []
    while True:
        _, _, terminated, truncated, _ = env.step(expert.act(env.state))
        car = env.state.car
        poses.append((car.x, car.y, car.heading))
        if terminated or truncated:
            return env.track, np.array(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--no-gif", action="store_true")
    args = parse_args(parser)

    config = presets.agentic()
    results = {"metadata": run_metadata(args.seeds), "preset": "agentic", "config": config}
    for name, make_policy in [("random", RandomPolicy), ("expert", lambda seed: ExpertPolicy())]:
        records = [run_episode(config, make_policy(seed), seed)[0] for seed in args.seeds]
        results[name] = {"summary": summarize(records), "episodes": records}
        s = results[name]["summary"]
        print(f"{name:7s} vuelta media {s['lap_fraction_mean']:.2f}  vueltas {s['completed_laps']}/{s['episodes']}"
              f"  fines {s['end_reasons']}  return medio {s['return_mean']:.2f}")
    path = save_json("ch01_agent_loop.json", results)
    print(f"guardado en {path}")

    if not args.no_gif:
        track, poses = expert_lap_poses(args.seeds[0])
        gif_path = results_dir() / "ch01_expert_lap.gif"
        episode_gif(track, poses[::2], str(gif_path), fps=5)  # una de cada dos decisiones: GIF más liviano
        print(f"GIF en {gif_path}")


if __name__ == "__main__":
    main()
