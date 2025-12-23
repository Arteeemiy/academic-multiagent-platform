import subprocess
import os
from pathlib import Path

Path("docs/retrieval_eval.md").unlink(missing_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, check=True, env=env)


def main():
    run(["pytest", "tests/test_retriever.py::test_retriever_k_sweep"])
    run(["pytest", "tests/test_retriever.py::test_retriever_embedding_sweep"])
    run(["pytest", "tests/test_retriever.py::test_retriever_chunking_sweep"])


if __name__ == "__main__":
    main()
