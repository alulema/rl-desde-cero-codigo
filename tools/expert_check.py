"""Criterio de la Fase 1: ¿el experto da vueltas en el circuito fijo? 5 semillas × 3 agresiones, con
spawn con ruido, para cada modelo de agarre.

    uv run python tools/expert_check.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from common import run_metadata, save_json  # noqa: E402
from rlcar import presets  # noqa: E402
from rlcar.directive import AGGRESSION_LEVELS  # noqa: E402
from rlcar.evaluate import ExpertPolicy, evaluate, summarize  # noqa: E402

SEEDS = range(5)


def main():
    results = {"metadata": run_metadata(SEEDS), "runs": []}
    for grip_name, base in [("rotate", presets.agentic()), ("redirect", presets.agentic_grip()),
                            ("erase", presets.grip_bug())]:
        for aggression in AGGRESSION_LEVELS:
            summary = summarize(evaluate(replace(base, aggression=aggression), ExpertPolicy(), SEEDS))
            results["runs"].append({"grip": grip_name, "aggression": aggression, "summary": summary})
            print(f"{grip_name:8s} agresión {aggression}: vueltas {summary['completed_laps']}/{summary['episodes']}"
                  f"  tiempo {summary['lap_time_mean']:.1f} s  freno {summary['mean_brake']:.3f}"
                  f"  v {summary['mean_speed']:.1f} m/s  return {summary['return_mean']:.2f}")
    print(f"guardado en {save_json('expert_check.json', results)}")


if __name__ == "__main__":
    main()
