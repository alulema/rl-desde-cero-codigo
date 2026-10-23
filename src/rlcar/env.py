"""`RLCarEnv`: el entorno de Gymnasium que une pista, física, observación, recompensa y terminaciones.

Un `step()` es una decisión: 5 pasos de física a 50 Hz con la misma acción, como en Agentic Racing
(`DecisionPeriod = 5`). La recompensa y las terminaciones se evalúan en cada paso de física.

El estado privilegiado (pose, velocidad, proyección sobre la pista) está en `env.unwrapped.state`. Lo leen
el experto y el evaluador; el agente solo recibe la observación.
"""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlcar.directive import AGGRESSION_LEVELS
from rlcar.observations import ObsConfig, build_observation, observation_names
from rlcar.physics import CarParams, CarState
from rlcar.physics import step as physics_step
from rlcar.rewards import RewardConfig, end_terms, step_terms
from rlcar.state import DriveState
from rlcar.terminations import TerminationConfig, TerminationTracker
from rlcar.track import Track, fixed_circuit, random_circuit, technical_circuit


@dataclass(frozen=True)
class TrackConfig:
    kind: str = "fixed"  # "fixed" (fácil), "technical" (con horquillas) o "random"
    half_length: float = 120.0  # circuito fijo
    half_height: float = 60.0
    corner_radius: float = 40.0
    technical_straight: float = 200.0  # circuito técnico
    technical_hairpin_radius: float = 15.0
    width: float = 12.0
    random_seeds: tuple[int, ...] = (0,)  # pistas aleatorias: en cada reset se elige una de estas
    random_min_corner_radius: float | None = 30.0
    # Forma de las pistas aleatorias. Los valores por defecto son los suaves; random_tracks(aggressive=True)
    # usa los de Agentic Racing antes de suavizarse, que producen curvas muy cerradas.
    random_harmonics: tuple[int, int] = (1, 2)
    random_harmonic_freq: tuple[int, int] = (2, 3)
    random_harmonic_amp: tuple[float, float] = (0.08, 0.16)
    random_radial_jitter: float = 0.035


@dataclass(frozen=True)
class SpawnConfig:
    clean: bool = False  # True: centrado, alineado y a clean_speed, como el CleanSpawn del evaluador
    heading_noise_deg: float = 10.0
    lateral_noise: float = 2.0  # metros
    launch_speed: float = 8.0  # m/s a lo largo de la pista: nunca se arranca detenido
    clean_speed: float = 14.0


@dataclass(frozen=True)
class EnvConfig:
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarParams = field(default_factory=CarParams)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    physics_dt: float = 0.02
    decision_period: int = 5
    aggression: float | None = None  # None = se sortea en cada episodio entre AGGRESSION_LEVELS


class RLCarEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self._tracks: dict = {}
        self.track = self._build_track(self.config.track.random_seeds[0])

        n_obs = len(observation_names(self.config.obs))
        self.observation_space = gym.spaces.Box(-2.0, 2.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32)
        )
        self.state: DriveState | None = None
        self.tracker: TerminationTracker | None = None

    @property
    def observation_names(self) -> list[str]:
        return observation_names(self.config.obs)

    def _build_track(self, seed: int) -> Track:
        c = self.config.track
        if c.kind == "fixed":
            if "fixed" not in self._tracks:
                self._tracks["fixed"] = fixed_circuit(c.half_length, c.half_height, c.corner_radius, c.width)
            return self._tracks["fixed"]
        if c.kind == "technical":
            if "technical" not in self._tracks:
                self._tracks["technical"] = technical_circuit(c.technical_straight, c.technical_hairpin_radius,
                                                              c.width)
            return self._tracks["technical"]
        if seed not in self._tracks:
            self._tracks[seed] = random_circuit(
                seed,
                harmonics=c.random_harmonics,
                harmonic_freq=c.random_harmonic_freq,
                harmonic_amp=c.random_harmonic_amp,
                radial_jitter=c.random_radial_jitter,
                min_corner_radius=c.random_min_corner_radius,
                width=c.width,
            )
        return self._tracks[seed]

    # region: reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c = self.config
        rng = self.np_random
        if c.track.kind == "random":
            self.track = self._build_track(int(rng.choice(c.track.random_seeds)))
        track = self.track

        # Spawn en un punto al azar de la línea central, mirando en el sentido de la pista, con ruido.
        index = int(rng.integers(track.n))
        track_heading = track.segment_headings[index]
        if c.spawn.clean:
            heading, lateral, speed = track_heading, 0.0, c.spawn.clean_speed
        else:
            heading = track_heading + np.radians(rng.uniform(-1, 1) * c.spawn.heading_noise_deg)
            lateral = rng.uniform(-1, 1) * c.spawn.lateral_noise
            speed = c.spawn.launch_speed
        position = track.points[index] + lateral * np.array([-np.sin(heading), np.cos(heading)])
        # Como en el original, la velocidad inicial sigue a la pista, no a la trompa con ruido.
        velocity = speed * np.array([np.cos(track_heading), np.sin(track_heading)])
        car = CarState(position[0], position[1], heading, velocity[0], velocity[1])

        aggression = c.aggression if c.aggression is not None else float(rng.choice(AGGRESSION_LEVELS))
        self.state = DriveState(track=track, car=car, projection=track.project(car.position), aggression=aggression)
        self.tracker = TerminationTracker(c.termination)
        self._history = []  # observaciones anteriores, si se apilan
        return self._observation(), {"lap_fraction": 0.0, "time": 0.0}
    # endregion: reset

    # region: step
    def step(self, action):
        c = self.config
        action = np.clip(np.asarray(action, dtype=float), self.action_space.low, self.action_space.high)
        state, track = self.state, self.track
        reward_terms: dict[str, float] = {}
        end_reason = None

        for _ in range(c.decision_period):
            state.car = physics_step(state.car, action, c.physics_dt, c.car)
            previous_progress = state.projection.progress
            state.projection = track.project(state.car.position, hint=state.projection.index)
            forward_metres = track.forward_metres(previous_progress, state.projection.progress)
            state.time += c.physics_dt

            end_reason = self.tracker.update(state, forward_metres, c.physics_dt)
            terms = step_terms(state, forward_metres, self.tracker.stuck_active, c.physics_dt, c.reward,
                               c.car.max_speed)
            if end_reason:
                time_left = max(0.0, 1 - state.time / c.termination.time_limit_seconds)
                terms.update(end_terms(end_reason, time_left, c.reward))
            for name, value in terms.items():
                reward_terms[name] = reward_terms.get(name, 0.0) + float(value)
            if end_reason:
                break

        state.last_action = action
        terminated = end_reason is not None
        truncated = not terminated and state.time >= c.termination.time_limit_seconds - 1e-9
        if truncated:
            end_reason = "time_limit"
        reward = float(sum(reward_terms.values()))
        info = {
            "reward_terms": reward_terms,
            "end_reason": end_reason,
            "lap_fraction": state.lap_fraction,
            "time": state.time,
        }
        return self._observation(), reward, terminated, truncated, info
    # endregion: step

    def _observation(self) -> np.ndarray:
        current = build_observation(self.state, self.config.obs, self.config.car.max_speed)
        stack = self.config.obs.stack
        if stack == 1:
            return current
        # Al arrancar no hay pasado: se repite la primera observación.
        self._history = ([current] + self._history)[:stack] if self._history else [current] * stack
        return np.concatenate(self._history)

    def render(self):
        if self.render_mode == "rgb_array":
            from rlcar.render import render_state

            return render_state(self.state)
        return None
