import os
import subprocess

from pathlib import Path


def run_jupytext_notebook(
    notebook, str_in_output: str = ""
) -> subprocess.CompletedProcess:
    path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "jupytext_notebooks"
        / notebook
    )
    env = dict(MPLBACKEND="Agg", **os.environ)
    result = subprocess.run(["python", path], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert str_in_output in result.stdout
    return result


def test_tutorial_01_basics():
    run_jupytext_notebook("01_basics.py", "Solution vector: ")


def test_tutorial_02_himmelblau():
    run_jupytext_notebook("02_himmelblau.py", "# gradient evaluations: ")


def test_tutorial_03_mini_batch_regression():
    run_jupytext_notebook("03_mini_batch_regression.py", "Solution: ")
