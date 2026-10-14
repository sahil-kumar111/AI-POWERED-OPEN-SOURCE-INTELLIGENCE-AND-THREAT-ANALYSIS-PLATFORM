from pathlib import Path
import json
import numpy as np

output_dir = Path("representation_learning/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print("Output folder:", output_dir.resolve())
print("NumPy version:", np.__version__)
print("Output folder ready.")
print("Next: rerun the E5 demo and save its actual vectors.")
