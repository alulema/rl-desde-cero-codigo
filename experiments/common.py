"""Utilidades compartidas por los experimentos: dónde guardar y cómo dejar constancia de qué se corrió."""

import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOCAL = RESULTS / "local"
_output = RESULTS


def git_commit() -> str:
    """El commit actual, con `-dirty` si hay cambios sin commitear: un resultado tiene que poder rehacerse.

    Sin `git`, o fuera de un repositorio (por ejemplo, desde el ZIP), devuelve "desconocido".
    """
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "experiments", "pyproject.toml",
                                "uv.lock"], cwd=root, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "desconocido"
    if not commit:
        return "desconocido"
    return commit + ("-dirty" if dirty else "")


def _same(value, default) -> bool:
    if isinstance(value, list) and isinstance(default, list):
        return sorted(value) == sorted(default)
    return value == default


def parse_args(parser, selectors=()):
    """Lee los argumentos y decide dónde se guardan los resultados.

    Solo la corrida del libro escribe en `results/`: los mismos valores por defecto, salvo los
    `selectors`, que eligen qué resultado del libro se rehace (el circuito, el experto) sin cambiarlo.
    Cualquier otra corrida, por ejemplo una versión corta de un recuadro "Pruébalo", va a
    `results/local/`, que git ignora, para no pisar los resultados con los que se escribió el libro.
    """
    global _output
    args = parser.parse_args()
    defaults = vars(parser.parse_args([]))
    changed = [name for name, value in vars(args).items()
               if name not in selectors and name != "no_gif" and not _same(value, defaults[name])]
    _output = LOCAL if changed else RESULTS
    if changed:
        print(f"corrida distinta de la del libro ({', '.join(changed)}): se guarda en {LOCAL}", flush=True)
    return args


def results_dir() -> Path:
    """`results/` para la corrida del libro, `results/local/` para cualquier otra (ver `parse_args`)."""
    _output.mkdir(parents=True, exist_ok=True)
    return _output


def run_metadata(seeds) -> dict:
    return {
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "seeds": list(seeds),
        "book_run": _output == RESULTS,
        "commit": git_commit(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def to_jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"no se puede serializar {type(value)}")


def save_json(name: str, payload: dict) -> Path:
    path = results_dir() / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=to_jsonable) + "\n")
    return path
