"""
run_pipeline.py
Runs the full ML pipeline end-to-end:
    1. Generate synthetic data
    2. Train all models
    3. Evaluate and plot
    4. Run SHAP explainability

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os


def run(script):
    print(f"\n{'='*55}")
    print(f"  Running: {script}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # run("data/generate_data.py")
    run("src/train.py")
    run("src/evaluate.py")
    run("src/explain.py")

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("  Plots saved to: plots/")
    print("  Models saved to: models/")
    print("\n  Start the API with:")
    print("    uvicorn api.main:app --reload")
    print("  Then open: http://localhost:8000/docs")
    print("=" * 55)
