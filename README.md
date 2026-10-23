# Reinforcement Learning desde cero — código

Código complementario del libro *Reinforcement Learning desde cero*, de Alexis Alulema: el entorno
`rlcar`, los algoritmos (PPO, BC, DAgger, GAIL y sus combinaciones), un script por experimento, los
resultados que cita el libro y los tests.

Todo corre en una laptop, solo con CPU. No hace falta Unity, GPU ni una cuenta en la nube. El apéndice E
del libro explica la instalación con más detalle.

## Instalación

Con [uv](https://docs.astral.sh/uv/) (recomendado), desde la carpeta del repositorio:

```bash
uv sync          # crea el entorno con Python 3.12 y las versiones exactas de uv.lock
uv run pytest    # corre los tests: unos cien, en menos de un minuto
```

Con `pip` (Python 3.12 o 3.13; la 3.14 todavía no):

```bash
python3.12 -m venv .venv           # en Windows: py -3.12 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -e . pytest
pytest
```

En Linux, para no bajar la versión de PyTorch con CUDA, instala la de CPU después de activar el entorno y
antes de `pip install -e . pytest`: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
`pip` no lee `uv.lock`; para los números exactos del libro, usa uv.

## Primeros pasos con `rlcar`

```python
from collections import Counter

import rlcar
from rlcar import presets
from rlcar.expert import Expert

env = rlcar.RLCarEnv(presets.agentic())      # o gymnasium.make("rlcar/RLCar-v0", config=...)
observation, info = env.reset(seed=0)
expert = Expert()
totals = Counter()
while True:
    action = expert.act(env.state)          # el experto lee el estado real; un agente, la observación
    observation, reward, terminated, truncated, info = env.step(action)
    totals.update(info["reward_terms"])      # los términos de este paso; sumados, los del episodio
    if terminated or truncated:
        break
print(info["end_reason"], dict(totals))
```

## Los experimentos

Un script por experimento del libro, en `experiments/`. Estos son los comandos que produjeron los
resultados de `results/`, y lo que tardan en la laptop donde se escribió el libro:

| Capítulo | Comando | Tiempo |
|---|---|---|
| 0 | `uv run python experiments/ch00_ml_minimo.py` | segundos |
| 1 | `uv run python experiments/ch01_agent_loop.py --no-gif` | segundos |
| 2 | `uv run python experiments/ch02_ppo.py --circuit easy` (y `--circuit technical`) | 6 min por circuito |
| 2 | `uv run python experiments/ch02_sb3_baseline.py --circuit easy` (y `technical`) | 4 min por circuito |
| 2 | `uv run python experiments/ch02_ablations.py` | 4–8 min por variante |
| 3 | `uv run python experiments/ch03_bc_dagger.py --expert memoryless` (y `memory`) | 2,5 min por semilla |
| 4 | `uv run python experiments/ch04_gail.py` | 90 s por semilla y variante |
| 5 | `uv run python experiments/ch05_combine.py` | una hora |
| 6 | `uv run python experiments/ch06_rewards.py` | 45 min |
| 7 | `uv run python experiments/ch07_observations.py` | dos horas |
| 8 | `uv run python experiments/ch08_debugging.py` | 45 min |
| 8 | `uv run python experiments/ch08_noise.py` | 40 min |
| 0–7 | `uv run python experiments/exercises.py` | 10 min |

Los scripts de entrenamiento aceptan `--seeds` (por defecto 0 a 4; 0 a 2 en `ch02_sb3_baseline.py` y
`ch02_ablations.py`) y `--steps`. `ch00_ml_minimo.py` y `exercises.py` aceptan una sola semilla, con
`--seed`.

**Solo la corrida del libro escribe en `results/`.** Con cualquier otra opción, por ejemplo `--seeds 0`
para una versión corta, el script lo avisa y guarda en `results/local/`, que git ignora. Cada JSON de
`results/` tiene un bloque `metadata` con el comando, las semillas, el commit y la fecha: para comparar,
corre el mismo comando y compara todo menos los tiempos, la fecha y el commit.

## Estructura

- `src/rlcar/`: el entorno de juguete, un auto 2D en una pista con la API de Gymnasium: pista, física,
  observación, recompensa, terminaciones, experto, evaluador, render, la directiva y los chequeos de
  depuración.
- `src/algos/`: los ejemplos del capítulo 0, el return y GAE, PPO, BC, DAgger, GAIL y sus combinaciones,
  escritos para leerse.
- `experiments/`: un script por experimento del libro.
- `results/`: los resultados que cita el libro.
- `tests/`: los tests, incluidos los ejemplos resueltos y las soluciones de los ejercicios.
- `tools/`: `expert_check.py`, que comprueba que el experto da vueltas en el circuito fijo, y `bench_rlcar.py`,
  que mide cuántas decisiones por segundo da el entorno en tu máquina.
- `docs/rlcar-design.md`: el diseño de `rlcar` y su tabla de fidelidad con Agentic Racing.

## Licencia

El código está bajo licencia MIT (ver `LICENSE`). El texto del libro, las figuras y la portada no son
libres: © Alexis Alulema, todos los derechos reservados.
