from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np


def fit_pchip(
    df: pd.DataFrame, x_col: str, y_col: str, num_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to a dataframe.
    PCHIP preserves monotonicity and is shape-preserving.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with x and y columns
    x_col : str
        Name of the x column (default: 'Displacement')
    y_col : str
        Name of the y column (default: 'Energy')
    num_points : int
        Number of points for smooth interpolation (default: 200)

    Returns:
    --------
    x_smooth : numpy.ndarray
        Smooth x values
    y_smooth : numpy.ndarray
        Smooth y values from PCHIP interpolation
    """
    # Extract x and y values
    x = df[x_col].values
    y = df[y_col].values

    # Create PCHIP interpolator
    pchip = PchipInterpolator(x, y)

    # Generate smooth x values
    x_smooth = np.linspace(x.min(), x.max(), num_points)

    # Evaluate PCHIP at smooth x values
    y_smooth = pchip(x_smooth)

    return x_smooth, y_smooth
