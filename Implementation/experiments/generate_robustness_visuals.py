import os
import sys

sys.path.append(os.getcwd())
from experiments.generate_qualitative_suite import run_qualitative_inference

if __name__ == "__main__":
    run_qualitative_inference()
