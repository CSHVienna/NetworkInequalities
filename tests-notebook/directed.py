"""
Add nbmake to automatically execute notebooks during test runs and catch errors.
"""

import os
import unittest
from pathlib import Path

import nbmake
from nbmake.preprocessors import ExecutePreprocessor

from netin.graphs import directed


class TestNotebooks(unittest.TestCase):
    """
    Test cases for the notebooks.
    """

    def run_notebook(self, directed_ipynb):
    
        root_path = Path(__file__).parent.parent
        notebook_path = (
            root_path / "docs" / "source" / "tutorials" / f"{directed}.ipynb"
        )

        # Store the current directory
        original_dir = Path.cwd()

        # Change to the directory containing the notebook
        notebook_dir = notebook_path.parent
        os.chdir(notebook_dir)

        # Load the notebook
        with open(notebook_path.name, "r", encoding="utf-8") as notebook_file:
            notebook_content = nbmake.read(notebook_file, as_version=4)

        # Initialize the ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=1000, kernel_name="python3")

        # Execute the notebook
        ep.preprocess(notebook_content, {"metadata": {"path": "."}})

        # Return to the original directory
        os.chdir(original_dir)

    def test_CRep_notebook_execution(self):
        self.run_notebook("CRep")

    def test_JointCRep_notebook_execution(self):
        self.run_notebook("JointCRep")

    def test_MTCOV_notebook_execution(self):
        self.run_notebook("MTCOV")

    def test_DynCRep_notebook_execution(self):
        self.run_notebook("DynCRep")

    def test_ACD_notebook_execution(self):
        self.run_notebook("ACD")

    def test_unknown_structure_notebook_execution(self):
        self.run_notebook("Unknown_structure")