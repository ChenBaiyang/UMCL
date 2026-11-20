from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TorchRuntimeEnvironmentTests(unittest.TestCase):
    def test_import_sets_cublas_workspace_before_runtime_calls(self) -> None:
        environment = dict(os.environ)
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; import utils.torch_runtime; "
                    "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''))"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), ":4096:8")

    def test_import_preserves_explicit_supported_value(self) -> None:
        environment = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; import utils.torch_runtime; "
                    "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''))"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), ":16:8")


if __name__ == "__main__":
    unittest.main()
