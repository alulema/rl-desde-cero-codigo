"""La directiva del estratega, reducida a un número: la agresión (0..1).

En Agentic Racing la directiva tenía 6 canales. Aquí queda la agresión, con el mismo mapeo de
`StrategyDirectiveMap`: más agresión = más velocidad objetivo y frenar más tarde.
"""

from rlcar.physics import lerp

# Los niveles bajo / medio / alto de `StrategyDirectiveMap.levelValues`.
AGGRESSION_LEVELS = (0.15, 0.5, 0.85)


def speed_scale(aggression: float) -> float:
    """Multiplicador de la velocidad objetivo (0.86 a 1.16)."""
    return lerp(0.86, 1.16, aggression)


def brake_margin(aggression: float) -> float:
    """m/s por encima del objetivo antes de frenar (0.3 a 2.2)."""
    return lerp(0.3, 2.2, aggression)
