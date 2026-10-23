"""Capítulo 0: aprender el volante del experto con descenso de gradiente.

El experto sin memoria gira el volante con una fórmula escrita a mano:

    volante = recortar((ángulo al punto de mira − 1.4 × desplazamiento lateral) / 16, −1, 1)

con el ángulo en grados y el desplazamiento en metros (y la corrección lateral recortada a ±13°). Mientras
nada se recorta, es un modelo lineal con dos pesos conocidos: 1/16 = 0.0625 y −1.4/16 = −0.0875. Este
experimento junta datos del experto en el circuito técnico y comprueba:

1. Que el descenso de gradiente encuentra esos dos pesos a partir de los datos, y cuánto cuesta con las
   entradas en sus unidades originales frente a estandarizadas: con varias tasas de aprendizaje, cuántos
   pasos tarda en bajar el error de 1e-6 y con cuál se dispara. La tasa máxima que no se dispara se
   calcula también de forma exacta (2 dividido por el mayor autovalor de la curvatura del error).
2. Cuánto más ruidoso es el camino con minibatches que con todos los datos.
3. Que con todos los datos (incluidos los recortados) un modelo lineal ya no alcanza, y una red pequeña
   sí se acerca.
4. Cómo se ve una normal de media 0.3 y desviación 0.1 cuando se muestrea.

    uv run python experiments/ch00_ml_minimo.py --seed 0
"""

import argparse

import numpy as np
import torch
from torch import nn

from common import parse_args, run_metadata, save_json
from algos.ml_minimo import gradient_descent, mse, predict, standardize, unstandardize_weights
from algos.ppo import ActorCritic, PPOConfig, mlp
from rlcar import RLCarEnv, presets
from rlcar.expert import Expert

EPISODE_SEEDS = range(20)
TRUE_WEIGHTS = [1 / 16, -1.4 / 16]
LEARNING_RATES = {"raw": [0.001, 0.01, 0.019, 0.021], "standardized": [0.1, 0.4, 0.49, 0.52]}
STEPS = 5000
TARGET_LOSS = 1e-6


def collect(seeds) -> dict:
    """Una fila por decisión: el ángulo al punto de mira, el desplazamiento lateral y el volante."""
    config = presets.technical()
    rows = []
    for seed in seeds:
        env, expert = RLCarEnv(config), Expert(memory=False)
        env.reset(seed=seed)
        while True:
            state = env.state
            inputs = [expert.aim_error(state), state.projection.lateral]
            action = expert.act(state)
            if state.car.forward_speed >= 1.5:  # por debajo, el experto usa otra fórmula de arranque
                rows.append(inputs + [action[0]])
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    data = np.array(rows)
    x, y = data[:, :2], data[:, 2]
    linear = (np.abs(y) < 1) & (np.abs(1.4 * x[:, 1]) < 13)
    return {"x": x, "y": y, "linear": linear}


def every(history, n=10):
    return [float(v) if np.isfinite(v) else None for v in history[::n]]


def max_learning_rate(x) -> float:
    """La tasa a partir de la cual el descenso de gradiente se dispara: 2 / el mayor autovalor de la
    curvatura del error cuadrático, 2 Xᵀ X / N (con una columna de unos por el sesgo)."""
    xb = np.c_[x, np.ones(len(x))]
    return float(2 / np.linalg.eigvalsh(2 * xb.T @ xb / len(x)).max())


def descend(x, y, learning_rate) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        w, b, history = gradient_descent(x, y, learning_rate, STEPS)
    reached = next((i + 1 for i, v in enumerate(history) if v < TARGET_LOSS), None)
    diverged = not np.isfinite(history[-1]) or history[-1] > history[0]
    return {"w": w.tolist(), "b": b, "final_loss": None if diverged else history[-1],
            "steps_to_target": reached, "diverged": diverged, "history": every(history)}


def fit_mlp(x, y, seed, steps=5000, minibatch_size=256) -> tuple[float, int]:
    """Una red de 2 capas ocultas de 64, como las de PPO y BC en este libro, con Adam."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = mlp(x.shape[1], (64, 64), 1, output_gain=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(len(y), size=minibatch_size))
        loss = ((net(xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = ((net(xt).squeeze(-1) - yt) ** 2).mean().item()
    return final, sum(p.numel() for p in net.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parse_args(parser)
    results = {"metadata": run_metadata([args.seed]), "episode_seeds": list(EPISODE_SEEDS),
               "true_weights": TRUE_WEIGHTS}

    data = collect(EPISODE_SEEDS)
    x, y, linear = data["x"], data["y"], data["linear"]
    xl, yl = x[linear], y[linear]
    results["data"] = {"samples": len(y), "linear_samples": int(linear.sum()),
                       "aim_error_deg": {"mean": float(x[:, 0].mean()), "std": float(x[:, 0].std()),
                                         "min": float(x[:, 0].min()), "max": float(x[:, 0].max())},
                       "lateral_m": {"mean": float(x[:, 1].mean()), "std": float(x[:, 1].std()),
                                     "min": float(x[:, 1].min()), "max": float(x[:, 1].max())},
                       "steer_saturated": int((np.abs(y) >= 1).sum())}
    print(f"{len(y)} decisiones del experto; {linear.sum()} sin recorte", flush=True)

    # 1. Unidades originales contra entradas estandarizadas, con todos los datos sin recorte.
    xs, mean, std = standardize(xl)
    results["units"] = {"steps": STEPS, "target_loss": TARGET_LOSS, "mean": mean.tolist(), "std": std.tolist(),
                        "inputs_correlation": float(np.corrcoef(xl.T)[0, 1])}
    for name, inputs in (("raw", xl), ("standardized", xs)):
        runs = {}
        for lr in LEARNING_RATES[name]:
            run = descend(inputs, yl, lr)
            if name == "standardized" and not run["diverged"]:
                w, b = unstandardize_weights(np.array(run["w"]), run["b"], mean, std)
                run["w_original_units"], run["b_original_units"] = w.tolist(), b
            runs[str(lr)] = run
            print(f"{name}, tasa {lr}: error final {run['final_loss']}, pasos hasta {TARGET_LOSS}: "
                  f"{run['steps_to_target']}, pesos {run.get('w_original_units', run['w'])}", flush=True)
        results["units"][name] = {"max_learning_rate": max_learning_rate(inputs), "runs": runs}
        print(f"{name}: tasa máxima {results['units'][name]['max_learning_rate']:.4f}", flush=True)

    # 2. Minibatches de 32 contra todos los datos, mismas entradas estandarizadas.
    _, _, full = gradient_descent(xs, yl, 0.1, STEPS)
    _, _, mini = gradient_descent(xs, yl, 0.1, STEPS, minibatch_size=32, seed=args.seed)
    results["minibatch"] = {"size": 32, "learning_rate": 0.1, "final_loss": mini[-1], "full_final_loss": full[-1],
                            "history": [float(v) for v in mini[:200]], "full_history": [float(v) for v in full[:200]]}
    print(f"minibatch de 32: error final {mini[-1]:.3g}", flush=True)

    # 3. Todos los datos, con los recortes: lineal contra una red pequeña.
    xa, _, _ = standardize(x)
    w, b, history = gradient_descent(xa, y, 0.1, STEPS)
    mlp_loss, mlp_params = fit_mlp(xa, y, args.seed)
    actor = ActorCritic(18, 3, PPOConfig()).actor
    results["nonlinear"] = {"linear_loss": history[-1], "mlp_loss": mlp_loss, "mlp_parameters": mlp_params,
                            "ppo_actor_parameters": sum(p.numel() for p in actor.parameters()),
                            "zero_model_loss": mse(predict(np.zeros(2), 0.0, xa), y)}
    print(f"todos los datos: lineal {history[-1]:.4f}, red de 64×64 ({mlp_params} parámetros) {mlp_loss:.4f}, "
          f"predecir 0: {results['nonlinear']['zero_model_loss']:.4f}", flush=True)

    # 4. Una normal de media 0.3 y desviación 0.1, como la de una policy de PPO.
    samples = np.random.default_rng(args.seed).normal(0.3, 0.1, size=10_000)
    results["normal"] = {"mean": 0.3, "std": 0.1, "samples": 10_000,
                         "sample_mean": float(samples.mean()), "sample_std": float(samples.std()),
                         "within_1_std": float(np.mean(np.abs(samples - 0.3) < 0.1)),
                         "within_2_std": float(np.mean(np.abs(samples - 0.3) < 0.2)),
                         "first_five": samples[:5].tolist()}
    print(f"normal: {results['normal']}", flush=True)

    print(f"guardado en {save_json('ch00_ml_minimo.json', results)}")


if __name__ == "__main__":
    main()
