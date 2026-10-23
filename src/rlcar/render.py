"""Vista desde arriba de la pista y el auto, con matplotlib. Fondo claro: sirve para imprimir.

`render_state` devuelve una imagen (para `env.render()`); `episode_gif` corre un episodio y lo guarda como
GIF, para las figuras del libro.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from rlcar.state import DriveState  # noqa: E402
from rlcar.track import Track  # noqa: E402

TRACK_COLOR = "#d9d9d9"
EDGE_COLOR = "#555555"
CAR_COLOR = "#1f4e79"
TRAIL_COLOR = "#c0392b"
CAR_LENGTH, CAR_WIDTH = 4.2, 2.0  # metros, como la caja del auto en Agentic Racing


def draw_track(ax, track: Track) -> None:
    """La cinta de la pista, sus bordes, la línea central y la meta."""
    band = np.vstack([track.left_edge, track.left_edge[:1], track.right_edge[:1], track.right_edge[::-1]])
    ax.add_patch(Polygon(band, closed=True, facecolor=TRACK_COLOR, edgecolor="none"))
    for edge in (track.left_edge, track.right_edge):
        loop = np.vstack([edge, edge[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color=EDGE_COLOR, linewidth=1)
    center = np.vstack([track.points, track.points[:1]])
    ax.plot(center[:, 0], center[:, 1], color="white", linewidth=0.8, linestyle=(0, (6, 6)))
    ax.plot(*np.stack([track.left_edge[0], track.right_edge[0]]).T, color="black", linewidth=2)
    margin = track.width * 2
    ax.set_xlim(track.points[:, 0].min() - margin, track.points[:, 0].max() + margin)
    ax.set_ylim(track.points[:, 1].min() - margin, track.points[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.axis("off")


def car_outline(x: float, y: float, heading: float) -> np.ndarray:
    """Las cuatro esquinas del auto en el mundo."""
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [CAR_LENGTH / 2, CAR_WIDTH / 2]
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    return corners @ rotation.T + [x, y]


def render_state(state: DriveState, size: tuple[float, float] = (8, 5)) -> np.ndarray:
    fig, ax = plt.subplots(figsize=size, dpi=80)
    draw_track(ax, state.track)
    car = state.car
    # El auto es chico frente a la pista: se dibuja a escala ×3 para que se vea.
    ax.add_patch(Polygon(car_outline(0, 0, car.heading) * 3 + [car.x, car.y], facecolor=CAR_COLOR))
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def episode_gif(track: Track, poses: np.ndarray, path: str, fps: int = 10, car_scale: float = 3.0) -> None:
    """Guarda un GIF de un episodio. `poses` es un arreglo (pasos, 3) con x, y y rumbo por decisión.

    El auto se dibuja a escala `car_scale` para que se vea en una pista de cientos de metros.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    draw_track(ax, track)
    (trail,) = ax.plot([], [], color=TRAIL_COLOR, linewidth=1.5)
    car_patch = Polygon(np.zeros((4, 2)), facecolor=CAR_COLOR)
    ax.add_patch(car_patch)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)

    def update(i):
        x, y, heading = poses[i]
        trail.set_data(poses[: i + 1, 0], poses[: i + 1, 1])
        outline = car_outline(0, 0, heading) * car_scale + [x, y]
        car_patch.set_xy(outline)
        label.set_text(f"t = {i / fps:.1f} s")
        return trail, car_patch, label

    animation = FuncAnimation(fig, update, frames=len(poses), blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
