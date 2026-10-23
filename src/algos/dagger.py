"""DAgger (Ross, Gordon y Bagnell 2011), capítulo 3.

BC aprende solo de los estados que visita el experto. Cuando la policy se equivoca un poco, llega a
estados que el experto nunca visitó, y ahí se equivoca más: los errores se acumulan (covariate shift).
DAgger corrige eso con datos de los estados de la propia policy:

1. Entrena BC con las demostraciones del experto.
2. Deja manejar a la policy. En cada paso, pregúntale al experto qué habría hecho (sin ejecutarlo).
3. Suma esas etiquetas al conjunto de datos y vuelve a entrenar desde cero. Repite.
"""

from dataclasses import dataclass, replace
from typing import Callable

from algos.bc import BCConfig, BCPolicy, Demonstrations, Rollout, rollout, train_bc
from rlcar.env import EnvConfig
from rlcar.expert import Expert


@dataclass(frozen=True)
class DAggerConfig:
    iterations: int = 6
    episodes_per_iteration: int = 5
    bc: BCConfig = BCConfig()


# region: dagger
def dagger(env_config: EnvConfig, expert: Expert, demos: Demonstrations,
           config: DAggerConfig,
           episode_seeds: Callable[[int], list[int]],
           on_iteration=None) -> BCPolicy:
    """`episode_seeds(i)` da las semillas de los episodios de la iteración
    i. `on_iteration(i, policy, rollouts, data)` se llama después de cada
    reentrenamiento, para evaluar o registrar."""
    data = demos
    policy = train_bc(data, config.bc)
    if on_iteration:
        on_iteration(0, policy, [], data)
    for i in range(1, config.iterations + 1):
        # Maneja la policy; el experto, en la sombra, etiqueta cada paso.
        rollouts: list[Rollout] = [
            rollout(env_config, expert, seed, driver=policy.act)
            for seed in episode_seeds(i)]
        for r in rollouts:
            data = data + r.demonstrations
        # Se reentrena desde cero sobre todos los datos juntos.
        bc_config = replace(config.bc, seed=config.bc.seed + i)
        policy = train_bc(data, bc_config)
        if on_iteration:
            on_iteration(i, policy, rollouts, data)
    return policy
# endregion: dagger
