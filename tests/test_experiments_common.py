"""Dónde guardan los experimentos: solo la corrida del libro escribe en results/ (apéndice E)."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import common  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", choices=["easy", "technical"], default="easy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--no-gif", action="store_true")
    return parser


@pytest.mark.parametrize("argv, local", [
    ([], False),
    (["--seeds", "4", "3", "2", "1", "0", "--steps", "500000"], False),
    (["--circuit", "technical", "--no-gif"], False),
    (["--seeds", "0"], True),
    (["--steps", "100000"], True),
])
def test_only_the_book_run_writes_to_results(monkeypatch, argv, local):
    monkeypatch.setattr(sys, "argv", ["script.py", *argv])
    common.parse_args(make_parser(), selectors=("circuit",))
    assert (common._output == common.LOCAL) is local
    assert common.run_metadata([0])["book_run"] is not local
    monkeypatch.setattr(common, "_output", common.RESULTS)


def test_git_commit_without_git(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(common.subprocess, "run", missing)
    assert common.git_commit() == "desconocido"
