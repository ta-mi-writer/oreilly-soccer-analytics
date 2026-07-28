# /// script
# dependencies = [
#     "marimo",
#     "matplotlib>=3.10.1",
#     "numpy>=2.2.3",
#     "pandas>=2.2.3",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import matplotlib
    import numpy as np
    import pandas as pd

    return matplotlib, np, pd, sys


@app.cell
def _(matplotlib, np, pd, sys):
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    return


if __name__ == "__main__":
    app.run()
