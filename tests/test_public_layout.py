"""Check routing and moved imports without starting scientific jobs."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
import evaluate


class PublicLayoutTests(unittest.TestCase):
    def test_training_options_are_forwarded_unchanged(self):
        with patch("train.run_script", return_value=0) as run:
            train.main(["56bus", "train", "--seed", "2602", "--run-id", "check"])
            run.assert_called_once_with(
                train.ENTRIES["56bus"],
                ["train", "--seed", "2602", "--run-id", "check"],
            )
        with patch("train.run_script", return_value=0) as run:
            train.main(["123bus", "--help"])
            run.assert_called_once_with(train.ENTRIES["123bus"], ["--help"])

    def test_all_entry_targets_exist(self):
        for relative in [*train.ENTRIES.values(), *evaluate.NOTEBOOKS.values(),
                         evaluate.OPERATION]:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_python_sources_and_local_import_targets(self):
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "src" or node.module.startswith("src.")
                    or node.module == "configs" or node.module.startswith("configs.")
                ):
                    target = ROOT.joinpath(*node.module.split("."))
                    self.assertTrue(
                        target.with_suffix(".py").is_file()
                        or (target / "__init__.py").is_file(),
                        f"{path}: {node.module}",
                    )

    def test_notebooks_resolve_repository_from_their_own_directory(self):
        previous = Path.cwd()
        try:
            for path in (ROOT / "experiments").rglob("*.ipynb"):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "".join(notebook["cells"][0]["source"])
                os.chdir(path.parent)
                exec(compile(source, str(path) + "#root_setup", "exec"), {})
                self.assertEqual(Path.cwd(), ROOT)
        finally:
            os.chdir(previous)

    def test_root_help_does_not_start_scientific_code(self):
        for name in ("train.py", "evaluate.py"):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / name), "--help"],
                cwd=ROOT.parent, capture_output=True, text=True, check=True,
            )
            self.assertIn("56bus", completed.stdout)
            self.assertIn("123bus", completed.stdout)


if __name__ == "__main__":
    unittest.main()
