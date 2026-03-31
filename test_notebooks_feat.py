from pathlib import Path
import pytest

NOTEBOOK_DIR = Path("examples/notebooks")

notebooks = list(NOTEBOOK_DIR.glob("*.ipynb"))

@pytest.mark.parametrize("notebook", notebooks)
def test_notebooks(notebook):
    pass