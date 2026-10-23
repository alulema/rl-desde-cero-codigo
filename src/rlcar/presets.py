"""Configuraciones con nombre. Cada función devuelve un `EnvConfig`; las que modifican aceptan otra como
base, así que se combinan: `grip_bug(race04())`.

Los pesos vienen de `RaceAgent.cs` de Agentic Racing: los actuales (race07/08) y los de race04, leídos del
código anterior al commit `af9684e`. Ver docs/rlcar-design.md §6.3.
"""

from dataclasses import replace

from rlcar.env import EnvConfig, TrackConfig
from rlcar.physics import with_agentic_grip, with_grip_bug
from rlcar.rewards import RewardConfig
from rlcar.terminations import TerminationConfig


# region: preset_agentic
def agentic() -> EnvConfig:
    """La recompensa y las terminaciones de race07/08, adaptadas a `rlcar`. Es la referencia."""
    reward = RewardConfig(
        progress_per_metre=0.02,
        target_speed_per_sec=0.25,
        line_follow_per_sec=0.05,
        slow_per_sec=0.15,
        edge_per_sec=0.6,
        stuck_per_sec=0.3,
        off_track=1.0,
        stuck=1.0,
        wrong_way=1.0,
        stall=1.0,
        lap_bonus=12.0,
        fast_lap_bonus=8.0,
    )
    return EnvConfig(reward=reward)
# endregion: preset_agentic


# region: preset_race04
def race04() -> EnvConfig:
    """La versión con la que corrió race04: detenerse termina el episodio
    con −1 a los 3 s.

    Con esta recompensa apareció el atajo: rodar unos metros, detenerse y
    aceptar el −1.
    """
    reward = RewardConfig(
        progress_per_metre=0.02,
        speed_per_sec=0.15,
        line_follow_per_sec=0.10,
        edge_per_sec=0.5,
        edge_ramp=False,
        time_per_step=0.0005,  # por paso de física: 0.025 por segundo
        off_track=1.0,
        stuck=1.0,
        wrong_way=1.0,
        lap_bonus=12.0,
        fast_lap_bonus=8.0,
    )
    termination = TerminationConfig(
        stuck_seconds=3.0,
        wrong_way_seconds=2.5,
        wrong_way_arm_metres=None,
        stall_seconds=None,  # el chequeo de estancamiento llegó después
    )
    return EnvConfig(reward=reward, termination=termination)
# endregion: preset_race04


def sparse(base: EnvConfig | None = None) -> EnvConfig:
    """Solo el bono de vuelta: nada guía al agente hasta que completa una."""
    base = base or agentic()
    return replace(base, reward=RewardConfig(lap_bonus=12.0))


def no_target_speed(base: EnvConfig | None = None) -> EnvConfig:
    """`agentic()` sin el término de velocidad objetivo ni el castigo por ir lento."""
    base = base or agentic()
    return replace(base, reward=replace(base.reward, target_speed_per_sec=0.0, slow_per_sec=0.0))


def with_directive(base: EnvConfig | None = None) -> EnvConfig:
    """La velocidad objetivo se escala con la agresión: la recompensa por fin lee la directiva."""
    base = base or agentic()
    return replace(base, reward=replace(base.reward, directive_scales_target=True))


def survival_bias(base: EnvConfig | None = None, alive_per_sec: float = 0.5) -> EnvConfig:
    """Una recompensa siempre positiva por seguir vivo, sin bono de vuelta: terminar la vuelta corta el
    pago. Es el sesgo de supervivencia que la recompensa de GAIL también tiene."""
    base = base or agentic()
    reward = RewardConfig(alive_per_sec=alive_per_sec, off_track=1.0, stuck=1.0, wrong_way=1.0, stall=1.0)
    return replace(base, reward=reward)


def agentic_grip(base: EnvConfig | None = None) -> EnvConfig:
    """El agarre de Agentic Racing tal cual: redirige la velocidad lateral sumando magnitudes, así que
    girar acelera el auto. `rlcar` usa por defecto un agarre que conserva la rapidez."""
    base = base or agentic()
    return replace(base, car=with_agentic_grip(base.car))


def grip_bug(base: EnvConfig | None = None) -> EnvConfig:
    """El agarre que borra la velocidad lateral: cada giro a velocidad frena el auto."""
    base = base or agentic()
    return replace(base, car=with_grip_bug(base.car))


def zero_filled_observation(base: EnvConfig | None = None, keep: int = 9) -> EnvConfig:
    """Bug plantado (capítulo 8): solo llegan los primeros `keep` valores (los 9 rayos); el resto, ceros."""
    base = base or agentic()
    return replace(base, obs=replace(base.obs, zero_fill_from=keep))


def short_time_limit(base: EnvConfig | None = None, seconds: float = 25.0) -> EnvConfig:
    """Bug plantado (capítulo 8): un límite de tiempo más corto que una vuelta del experto en el circuito
    técnico (≈ 30 s). En Agentic Racing eran 80 s contra una vuelta de ≈ 95 s: la misma proporción."""
    base = base or agentic()
    return replace(base, termination=replace(base.termination, time_limit_seconds=seconds))


def all_bugs(base: EnvConfig | None = None) -> EnvConfig:
    """Los tres bugs plantados del entorno juntos. El cuarto, el signo de la línea base, está en el experto."""
    return short_time_limit(zero_filled_observation(grip_bug(base)))


def technical(base: EnvConfig | None = None, straight: float = 200.0, hairpin_radius: float = 15.0) -> EnvConfig:
    """El circuito técnico: rectas largas y horquillas de 180°, donde hay que frenar antes de la curva."""
    base = base or agentic()
    track = replace(base.track, kind="technical", technical_straight=straight, technical_hairpin_radius=hairpin_radius)
    return replace(base, track=track)


def random_tracks(
    base: EnvConfig | None = None,
    seeds: tuple[int, ...] = tuple(range(8)),
    min_corner_radius: float | None = 30.0,
    aggressive: bool = False,
) -> EnvConfig:
    """Pistas aleatorias con semilla.

    Con `aggressive=True` y `min_corner_radius=None` salen curvas muy cerradas, a menudo cerca del
    radio de giro del auto (≈ 11.5 m a 20 m/s) o por debajo.
    """
    base = base or agentic()
    track = TrackConfig(kind="random", random_seeds=seeds, random_min_corner_radius=min_corner_radius,
                        width=base.track.width)
    if aggressive:
        track = replace(track, random_harmonics=(1, 3), random_harmonic_freq=(2, 5),
                        random_harmonic_amp=(0.12, 0.30), random_radial_jitter=0.06)
    return replace(base, track=track)
