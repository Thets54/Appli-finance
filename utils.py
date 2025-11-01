import numpy as np
import pandas as pd

def annual_days_to_years(days: int) -> float:
    return days / 365.25

def tenor_to_years(tenor: str) -> float:
    tenor = tenor.strip().lower()
    if tenor.endswith("y"):
        return float(tenor[:-1])
    if tenor.endswith("m"):
        return float(tenor[:-1]) / 12.0
    raise ValueError("Tenor inconnu (ex: '6M', '1Y').")
