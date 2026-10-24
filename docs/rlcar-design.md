# Diseño de `rlcar`: el entorno de juguete

Estado: **aprobado por el autor** (2026-10-18), incluida la decisión D1 (§2).

> Este es el documento de diseño que se escribió **antes** de programar `rlcar`, y se conserva como se
> aprobó. Las "fases" son las del plan del libro (la Fase 1 es el entorno; la 2, PPO; la 4, las ablaciones
> y la depuración), y "la bitácora" es la del manuscrito, que no está en este repositorio. Los valores
> finales están en el código (`src/rlcar/`) y en el libro.

Fuente: `alulema/agentic-racing` en `33e1f8d` (2026-10-10), en particular:

- `RaceAgent.cs`, `CarController.cs`, `VehicleConfig.cs`, `TrackProgress.cs`, `TrainingArena.cs`,
  `TrackGenerator.cs` y `StrategyDirectiveMap.cs`;
- `training/config/race_ppo.yaml`;
- `docs/Devlog.md`;
- el historial de git, para la versión de la recompensa con la que corrió race04 (el commit anterior a
  `af9684e`).

---

## 1. Objetivo

`rlcar` es Agentic Racing en Python puro, con la API de Gymnasium. Tiene que cumplir dos cosas a la vez, y
casi todo el diseño sale de mantenerlas en equilibrio:

1. **Ser fiel en lo que enseña.** Las mismas unidades (metros, m/s, 10 decisiones por segundo), las mismas
   observaciones, la misma estructura de recompensa y de terminaciones, y los mismos tipos de fallo, cada uno
   activable con un parámetro. Así, lo que el libro cuenta de Agentic Racing se puede reproducir y medir
   aquí.
2. **Ser resoluble en minutos con CPU.** PPO tiene que poder aprender a dar vueltas en el circuito fijo en
   menos de 10–15 minutos. Si un entorno es fiel pero nadie puede entrenarlo, el libro no tiene nada que
   mostrar.

Cuando las dos cosas chocan, gana la segunda, y la diferencia se documenta en la tabla de fidelidad (§10)
para que el libro la cuente.

---

## 2. Escala: la decisión más importante

Agentic Racing es grande para un laptop:

- **El circuito:** unos 2 km de perímetro y 12 m de ancho.
- **Una vuelta:** unos 95 s a ritmo, es decir, unas 950 decisiones.
- **El entrenamiento:** cada corrida usó de 4 a 20 millones de pasos.

Un PPO en un solo archivo de Python, sin GPU, no llega a eso en minutos. La palanca es **acortar el
horizonte**: menos decisiones por vuelta significa que la recompensa de terminar llega antes y que el
descuento la castiga menos.

**Propuesta:** mantener la física y las unidades de Agentic Racing, y achicar el circuito.

- **Circuito fijo:** el mismo rectángulo de esquinas redondeadas, a escala, con un perímetro de unos
  500–700 m (por ejemplo, semiejes de 120 × 60 m y esquinas de radio 40 m, unos 730 m). La vuelta del experto
  debería quedar en 25–40 s, unas 250–400 decisiones.
- **Ancho de pista:** 12 m, igual que el original.
- **Valores finales:** se calibran en la Fase 1 con el experto, buscando que haga vueltas limpias con
  margen, y se anotan en la bitácora. Luego se congelan antes de la Fase 2.

**Qué cambia para el libro:** cuando hable del caso real, el libro usa los números de Agentic Racing (2 km,
95 s, γ = 0.995 → 0.37 a 20 s), y cuando hable de `rlcar`, los suyos. Las cuentas de descuento siguen
valiendo porque la frecuencia de decisión es la misma, 10 Hz.

> **Decisión D1 (2026-10-18), aprobada:** circuito fijo más chico que el original, con la misma física y
> las mismas unidades. El libro usa dos juegos de números, los de Agentic Racing y los de `rlcar`.

> **Decisión D2 (2026-10-18), del autor: dos circuitos fijos.** En el circuito fácil (651 m, esquinas de
> 40 m), PPO aprendió en segundos a dar vueltas a fondo y sin frenar: con 12 m de ancho, el auto abre la
> trazada y las curvas de 90° se toman casi a cualquier velocidad. Por eso se agrega un **circuito
> técnico** (`presets.technical()`): un óvalo tipo estadio con rectas de 200 m y horquillas de 180° de
> 15 m de radio, 494 m en total. En una horquilla no se puede abrir la trazada, y entrar a fondo termina
> fuera de la pista. El circuito fácil queda para mostrar que PPO aprende cuando el entorno es resoluble
> (capítulo 2). El técnico, para la anticipación de la curva (capítulos 6 y 7).

---

## 3. API

```python
import gymnasium as gym
import rlcar

env = gym.make("rlcar/RLCar-v0", config=rlcar.presets.agentic())   # o RLCarEnv(config) directamente
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

info["reward_terms"]   # {"progress": 0.31, "target_speed": 0.02, ...}, la recompensa del paso por término
info["end_reason"]     # None, o "lap", "off_track", "stuck", "wrong_way", "stall", "time_limit"
info["lap_fraction"]   # avance neto en la vuelta, 0..1
env.unwrapped.state    # estado privilegiado (pose, velocidad, progreso); lo leen el experto y el evaluador
```

- **Configuración:** todo vive en un `EnvConfig` (dataclass) con sub-configuraciones: `TrackConfig`,
  `CarConfig`, `ObsConfig`, `RewardConfig`, `TerminationConfig` y `SpawnConfig`. Los presets con nombre
  (`rlcar.presets`) son funciones que devuelven un `EnvConfig`, y cada experimento guarda en `results/` la
  configuración completa que usó.
- **Paso:** un `step()` es **una decisión**, que equivale a 5 pasos de física a 50 Hz con la acción
  repetida, como `DecisionPeriod = 5` con `TakeActionsBetweenDecisions`. La recompensa y las terminaciones se
  evalúan en cada paso de física y se acumulan, igual que en el original.
- **Semillas:** `reset(seed=...)` fija `np_random`, que controla la pista aleatoria, el spawn y la directiva.
  Con la misma semilla y las mismas acciones, la trayectoria es idéntica, y hay un test que lo comprueba.
- **Paralelismo:** se usa `gymnasium.vector.SyncVectorEnv`, o `AsyncVectorEnv` si conviene. Para SB3, su
  propio `VecEnv`. No se escribe un entorno vectorizado propio salvo que la medición de la Fase 1 lo exija
  (§9).

---

## 4. Módulos

| Archivo | Contenido |
|---|---|
| `src/rlcar/track.py` | `Track`: línea central como polilínea muestreada cada ~1 m, ancho, longitud, tangentes, curvatura y `turn_deg(desde_m, hasta_m)`. Proyección de un punto (muestra más cercana, progreso de 0 a 1, desplazamiento lateral con signo) con ventana móvil, como `TrackProgress`. Constructores `fixed_circuit()` y `random_circuit(seed, ...)` |
| `src/rlcar/physics.py` | `CarParams`, `CarState` y una función pura `step(state, action, dt, params) -> state` |
| `src/rlcar/sensors.py` | Rayos contra los bordes, vectorizados en NumPy (intersección rayo–segmento) |
| `src/rlcar/observations.py` | `ObsConfig` y `build_observation(...)`, que también devuelve los nombres de cada componente |
| `src/rlcar/rewards.py` | `RewardConfig` y un término por función. Devuelve un dict `{término: valor}` |
| `src/rlcar/terminations.py` | `TerminationConfig` y los temporizadores (detenido, contrasentido, sin progreso) |
| `src/rlcar/env.py` | `RLCarEnv(gym.Env)`: arma todo lo anterior |
| `src/rlcar/presets.py` | Configuraciones con nombre (§7) |
| `src/rlcar/expert.py` | El experto heurístico, con y sin memoria (§8) |
| `src/rlcar/evaluate.py` | El evaluador independiente de la recompensa (§9) |
| `src/rlcar/render.py` | Render 2D con matplotlib, y exportación a GIF (con `PillowWriter`, sin dependencias nuevas) o a MP4 si hay ffmpeg |
| `src/rlcar/bugs.py` | Los bugs plantados del capítulo 8, como modificadores de la configuración o *wrappers* |

Las funciones de física, sensores, recompensa y terminaciones son **puras**: reciben el estado y la
configuración y devuelven valores. Así se testean sin crear el entorno, y el libro las incluye por región.

---

## 5. Física (`physics.py`)

Es una adaptación a 2D de `CarController` y `VehicleConfig`, con los mismos parámetros por defecto, para que
las cuentas del libro valgan en los dos lados.

- **Estado:** posición `(x, y)`, rumbo `θ` y velocidad como vector `(vx, vy)` en el marco del mundo.
- **Convención:** ángulos en radianes, antihorario positivo, con `steer > 0` = izquierda. Unity usa la
  convención contraria (negativo = izquierda); el libro lo dice una vez.
- **Tracción** (`ApplyDrive`):
  - Freno: `BrakeForce/Mass` = 15 m/s², opuesto a la velocidad longitudinal, y tiene prioridad sobre el
    acelerador.
  - Acelerador: `EngineForce/Mass` = 10 m/s² × `throttle`, hasta `MaxSpeed` = 55 m/s hacia adelante o
    `MaxReverseSpeed` = 12 m/s en reversa.
  - Sin acelerador: rozamiento de `CoastForce/Mass` ≈ 2.1 m/s².
  - Arrastre lineal 0.2, como el `linearDamping` de Unity: `v ← v / (1 + 0.2·dt)`.
- **Giro** (`ApplySteering`): velocidad angular = `steer` × 130 °/s × autoridad. La autoridad va de 0.25 a
  1, según la velocidad hasta 0.4 m/s, y después baja hasta 0.35 al llegar a `MaxSpeed`. En reversa el giro
  se invierte. Con esto, el radio mínimo de giro a 20 m/s es ≈ 11.5 m, la cifra que la bitácora usa para
  hablar de curvas imposibles. Un test lo comprueba.
- **Agarre lateral** (`ApplyLateralGrip`): `k = clamp(LateralGrip·dt, 0, 1)`, con `LateralGrip` = 7, y se
  quita `k·v_lat` de la velocidad lateral. Qué se hace con lo quitado lo decide `grip_model`:
  - `"rotate"`, **el valor por defecto** (decisión del autor del 2026-10-18): la velocidad gira hacia la
    trompa sin cambiar de módulo.
  - `"redirect"`, el de Agentic Racing tras el arreglo (preset `agentic_grip()`): devuelve
    `GripRedirect·|Δ|` = 0.85·|Δ| hacia adelante. Como suma magnitudes en lugar de rotar el vector,
    **girar crea velocidad**: en `rlcar`, a 20 m/s en una curva de 40 m y sin acelerar, el auto termina a
    22.5 m/s al cabo de 2 s, cuando debería bajar a 10 m/s (lo mismo que en una recta). La fórmula es la de
    `CarController.cs`, así que muy probablemente Agentic Racing tenía este efecto, pero no se verificó en
    Unity.
  - `"erase"`, el bug anterior al arreglo (preset `grip_bug()`, con `LateralGrip` = 9): borra la
    velocidad lateral, y cada giro a velocidad frena el auto.
- **Sin muros:** el original tenía muros con colisión, y aquí no hay. Salirse más de 2 m del borde termina
  el episodio, igual que el `offTrackMargin` original. Así se evita un motor de colisiones; la diferencia
  está en §10.

---

## 6. Observación, acción y recompensa

### 6.1 Observación (`ObsConfig`)

Cada bloque se activa por separado, para las ablaciones del capítulo 7. El preset `agentic` reproduce el
vector original de 42 números, con dos diferencias marcadas.

| Bloque | Tamaño | Original | Nota |
|---|---:|---|---|
| Rayos | 9 por defecto | 9 rayos en 150°, de 70 m. ML-Agents da 3 números por rayo: 27 | `rlcar` da 1 número por rayo (distancia normalizada), o 3 con `ray_format="mlagents"` |
| Velocidad longitudinal y lateral | 2 | Igual, dividido por `MaxSpeed` | |
| Error de rumbo | 1 | Respecto a la **línea de carrera** | Aquí, respecto a la **línea central**. No hay línea de carrera (§10) |
| Desplazamiento lateral | 1 | Igual, en medios anchos y recortado a ±2 | |
| Desplazamiento de la línea de carrera | 1 | Igual | Sin línea de carrera vale siempre 0. Se omite, salvo en `agentic_exact` |
| Progreso en la vuelta (0–1) | 1 | Igual (`Distance01`) | Se puede apagar para la ablación de memorización |
| Curvatura por delante | 3 | Giro de la línea central en 0–22, 18–45 y 40–75 m, en grados / 90 | Las mismas ventanas: dependen de la velocidad y la frenada, no del tamaño de la pista |
| Directiva | 1 | 6 (agresión, riesgo y un *one-hot* de 4 tipos) | Simplificada a la agresión (§6.4) |
| Volante anterior | 1 | No existía | Opcional, para el capítulo 7 (memoria) |
| Apilado de *k* observaciones | ×k | `NumStackedVectorObservations = 1` | Con un *wrapper*, para el capítulo 7 |

Todas las observaciones quedan aproximadamente en [−1, 1] o [−2, 2]. La normalización móvil
(`normalize: true` en ML-Agents) la hace el algoritmo, no el entorno, porque el capítulo 2 la enseña.

### 6.2 Acción

`Box(low=[−1, −1, 0], high=[1, 1, 1])`: `steer`, `throttle` y `brake`. El entorno recorta lo que llegue
fuera de rango, igual que hacía `OnActionReceived`.

### 6.3 Recompensa (`RewardConfig`)

Cada término es una función con su peso, y `info["reward_terms"]` los devuelve **por separado**. Un término
con peso 0 no se calcula. Los nombres y unidades son los del original, por metro, por segundo o por evento,
porque el capítulo 6 enseña justamente eso.

| Término | Unidad | Original, race07/08 | Original, race04 |
|---|---|---:|---:|
| `progress` | por metro | 0.02 | 0.02 |
| `target_speed` (pico en la velocidad objetivo según la curva) | por segundo | 0.25 | — |
| `speed` (proporcional a `v/MaxSpeed`) | por segundo | — | 0.15 |
| `line_follow` (alineado y cerca de la línea) | por segundo | 0.05 | 0.10 |
| `slow` (por debajo del objetivo) | por segundo | 0.15 | — |
| `edge` (rampa desde 0.55 del medio ancho) | por segundo | 0.6 | 0.5 (sin rampa) |
| `time` (penalización por paso de física) | por paso | — | 0.0005 |
| `stuck` (detenido) | por segundo | 0.3 | — (era un evento de −1 que terminaba el episodio) |
| `off_track`, `stall`, `wrong_way` | evento | −1 | −1 |
| `lap_bonus` + `fast_lap_bonus` × presupuesto restante | evento | 12 + 8 | 12 + 8 |
| `alive` (constante por paso, para el sesgo de supervivencia) | por segundo | — | — |
| `directive_speed` (objetivo × `SpeedScale` de la directiva) | — | No existía | Lo propuso el post 6, sin probar |

`line_follow` usa la línea central, porque no hay línea de carrera. `wall_hit` desaparece, porque no hay
muros.

### 6.4 Directiva

Se reduce a un escalar: la **agresión**, que se sortea por episodio entre {0.15, 0.5, 0.85}, los mismos
niveles que `levelValues`. Se mapea como en `StrategyDirectiveMap`:

- `SpeedScale` va de 0.86 a 1.16;
- `BrakeMargin` va de 0.3 a 2.2 m/s.

El experto la obedece, y la recompensa la ignora salvo en el preset `with_directive`. Ese es el punto ciego
del capítulo 6 y el experimento "atacar frente a conservar" del capítulo 5.

**Simplificación:** sin tipos (*attack*, *defend*, ...) ni tolerancia al riesgo. En el original, esos dos
canales movían sobre todo el sesgo hacia la línea de carrera (`CentreBlend`), que aquí no existe.

---

## 7. Episodios, terminaciones y presets

### 7.1 Episodio

Un episodio es **una vuelta**, como en el original:

- **Spawn:** en un punto al azar de la línea central, con ruido de ±10° en el rumbo y ±2 m en lo lateral, y
  arranque lanzado a 8 m/s.
- **Spawn limpio:** con `clean_spawn`, el auto arranca centrado, alineado y a 14 m/s, como el `CleanSpawn`
  del evaluador.

### 7.2 Terminaciones

Todas son configurables. Las que terminan el episodio por un fallo o por completar la vuelta dan
`terminated`; el límite de tiempo da `truncated`.

| Motivo | Condición, original actual | Tipo |
|---|---|---|
| `lap` | Avanzó el 99 % del largo de la vuelta | terminated |
| `off_track` | \|lateral\| > medio ancho + 2 m | terminated |
| `stuck` | Detenido (< 0.5 m/s) más de 8 s, y solo después de haber arrancado | terminated |
| `wrong_way` | En contrasentido más de 4.5 s, y solo después de 15 m de progreso | terminated |
| `stall` | Menos de 8 m de progreso en 5 s | terminated |
| `time_limit` | Se acabó el tiempo (el original tenía 4000 pasos de física, 80 s, en race01–08) | **truncated** |

### 7.3 Presets

| Preset | Para qué |
|---|---|
| `agentic()` | La configuración de race07/08 adaptada (§6). Es la referencia |
| `race04()` | La recompensa y las terminaciones de race04: `stuck` termina con −1 a los 3 s, hay penalización por paso, y existen los términos `speed` y `line_follow` de entonces. Sirve para **reproducir el atajo** |
| `sparse()` | Solo `lap_bonus` y las terminaciones |
| `no_target_speed()` | `agentic()` sin el término de velocidad objetivo |
| `with_directive()` | `agentic()` con la velocidad objetivo escalada por la directiva |
| `survival_bias()` | `alive` > 0 y sin bono de vuelta: una recompensa siempre positiva |
| `grip_bug()` | `grip_redirect = 0` y `LateralGrip = 9` |
| `technical()` | El circuito técnico, con horquillas (D2) |
| `agentic_grip()` | El agarre de Agentic Racing, que crea velocidad al girar |
| `random_tracks(min_corner_radius=..., aggressive=...)` | Pistas al azar; con `aggressive=True` y sin filtro de radio aparecen curvas imposibles |

Los presets se pueden combinar, por ejemplo `grip_bug(race04())`.

---

## 8. El experto (`expert.py`)

Es una adaptación de `RaceAgent.Heuristic()`. Lee el **estado privilegiado** (`env.unwrapped.state`, que incluye la pista),
no la observación:

- **Velocidad objetivo:** se calcula con el giro más cerrado de la línea central en los próximos ~60 m
  (`lerp(0.40, 0.10)·MaxSpeed` según giro/45°) y se multiplica por `SpeedScale` de la directiva.
- **Acelerador y freno:**
  - por debajo del objetivo, acelerador a fondo;
  - por encima del objetivo más `BrakeMargin`, frena proporcionalmente;
  - entre los dos, acelerador de crucero.
- **Dirección:** persecución pura (*pure pursuit*) hacia un punto entre 12 y 36 m adelante en la línea
  central, más una corrección por el desplazamiento lateral.

Hay dos variantes:

- `Expert(memory=True)`: tiene el volante filtrado (`steer ← lerp(steer, raw, 0.35)`) y la reversa de escape
  con temporizador, que avanza con el tiempo real entre decisiones (el bug corregido del post 7).
- `Expert(memory=False)`: sin filtro ni temporizadores. Es una función pura del estado, y sirve de
  referencia para el capítulo 3, porque es lo que una policy sin memoria sí puede imitar.

**DAgger:** el experto tiene `reset()` y `act(state)`, y **se llama en cada paso**, también cuando la acción
la ejecuta la policy, para que su filtro y sus temporizadores sigan la trayectoria real. El script de DAgger
lo corre en la sombra, nunca como una consulta suelta sobre estados sueltos.

---

## 9. Evaluador, render y rendimiento

- **Evaluador (`evaluate.py`):** corre una policy (cualquier función de observación a acción, el experto o
  un modelo de SB3) durante N episodios con semillas fijas. No usa la recompensa para nada. Reporta:
  - fracción de vuelta, vueltas completas y tiempo por vuelta;
  - el reparto de motivos de fin;
  - velocidad, acelerador, freno y volante medios;
  - un registro por episodio y por segundo en CSV.

  Es la herramienta del capítulo 8, y también la que muestra el promedio engañoso, como los 5.3 m/s de la
  bitácora.
- **Render:** una vista desde arriba de la pista y el auto, con la estela y los rayos opcionales. Exporta
  GIF o MP4 de un episodio. Las figuras del libro salen de aquí, en versión clara desde el principio.
- **Rendimiento:** en la Fase 1 se mide cuántas decisiones por segundo da un entorno en tu Mac, y cuántas
  con 8 en paralelo. La meta orientativa es que PPO pueda ver unos 1–2 M de decisiones en menos de 15
  minutos, lo que exige unas 2 000 decisiones por segundo en total. Si no llega, primero se perfila (los
  rayos son el sospechoso), luego se vectoriza el sensor y, como último recurso, se escribe un entorno por
  lotes. La cifra medida va a la bitácora.

---

## 10. Tabla de fidelidad

Lo que el libro tiene que decir cuando compare `rlcar` con Agentic Racing:

| Aspecto | Agentic Racing | `rlcar` |
|---|---|---|
| Motor | Unity 6 + ML-Agents 4.0.3 | Python, NumPy y Gymnasium |
| Física | `CarController` con Rigidbody | Los mismos parámetros y ecuaciones, en 2D y sin motor de colisiones, salvo el agarre (abajo) |
| Agarre lateral | Redirige sumando magnitudes: girar crea velocidad | Por defecto conserva la rapidez; el original está en `agentic_grip()` |
| Muros | Colisión con rebote, más fin al salirse 2 m del borde | Sin muros, solo el fin al salirse 2 m del borde |
| Circuito fijo | ~2 km, vuelta de ~95 s | Dos: el fácil de 651 m (semiejes 120 × 60 m, esquinas de 40 m; el experto tarda 28–36 s) y el técnico de 494 m (rectas de 200 m, horquillas de 15 m; el experto tarda 25–32 s) |
| Línea de carrera | La había (`BuildRacingLine`) y se usaba en la observación y en `line_follow` | No la hay: se usa la línea central |
| Rayos | 3 números por rayo (formato ML-Agents) | 1 por defecto, 3 opcional |
| Directiva | 6 canales | 1 (la agresión) |
| Decisión | 10 Hz, física a 50 Hz | Igual |

---

## 11. Tests (pytest)

- **Pista:** longitud del circuito fijo contra la fórmula, proyección de puntos conocidos (progreso,
  desplazamiento lateral con signo, salto de 1 a 0 en la meta) y curvatura de un arco de radio conocido.
- **Física:**
  - aceleración desde parado hasta el tope;
  - distancia de frenado contra la cuenta;
  - radio de giro a 20 m/s ≈ 11.5 m;
  - con `grip_redirect = 0`, girar a velocidad pierde rapidez, y con 0.85 casi no.
- **Rayos:** distancias en una recta, que se pueden calcular a mano.
- **Recompensa:**
  - cada término por separado en estados construidos a mano;
  - la suma de `reward_terms` es igual a la recompensa;
  - un término con peso 0 no aparece.
- **Terminaciones:** cada motivo con una secuencia de estados construida, y `truncated` frente a
  `terminated`.
- **Entorno:**
  - `gymnasium.utils.env_checker.check_env`;
  - determinismo con la misma semilla;
  - forma y nombres de la observación según `ObsConfig`.
- **Experto:** completa una vuelta en el circuito fijo con spawn limpio y con las tres agresiones, y el
  tiempo de vuelta baja al subir la agresión. Esto último es una expectativa: si no se cumple, se investiga
  y se cuenta.

---

## 12. Criterio de cierre de la Fase 1

Lo que pide el plan del libro, más lo que este diseño agrega:

1. El experto completa vueltas en el circuito fijo. Se miden el porcentaje de vueltas y el tiempo por vuelta
   con 5 semillas y las tres agresiones, y se anotan en la bitácora.
2. Los tests pasan, local y en CI.
3. Un episodio del experto exportado a GIF.
4. El rendimiento medido y anotado (§9).
5. Cada preset de fallo **existe y se comprueba mecánicamente**. Por ejemplo, con `race04()`, detenerse a
   los ~X m da un return que se calcula término por término. Que PPO *encuentre* esos fallos se verá en las
   Fases 2 y 4; aquí no se promete.

---

## 13. Discrepancias encontradas en la fuente

Hay que resolverlas antes de escribir los capítulos que las usan:

- **El return del atajo de la cuarta corrida (≈ +3.8).** La bitácora (2026-09-06) lo calcula como
  `245 m × 0.02 − 0.08 − 1`. Pero en el código con el que corrió race04 (el anterior a `af9684e`):
  - `OnActionReceived` corre en cada paso de física, así que `timePenaltyPerStep = 0.0005` equivale a
    0.025/s, y en un episodio de unos 13–15 s suma 0.3–0.4, no 0.08;
  - estaban activos `speed` (0.15/s × `v/MaxSpeed`) y `line_follow` (0.10/s), que la cuenta omite;
  - la misma tabla de la bitácora da un *plateau* de ~4.8 para race04, no de 3.8.

  El mecanismo, detenerse y cobrar el −1, está bien documentado. La cifra exacta, no. Con el preset
  `race04()` se puede sumar término por término la recompensa de esa política en `rlcar`, siempre aclarando
  que es `rlcar` y no Agentic Racing.
- **El 82 % de la heurística** fue su mejor episodio de 24 (18 terminaron `stuck`), no su promedio. La
  bitácora ya lo aclara en su entrada del 2026-10-02.
