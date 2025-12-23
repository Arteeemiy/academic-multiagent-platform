import subprocess


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    run(["pytest", "tests/test_llm.py::test_llm_model_sweep"])


if __name__ == "__main__":
    main()
