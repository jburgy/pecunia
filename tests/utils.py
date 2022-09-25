from math import erf

import numpy as np

SQRT2 = 2.0**0.5


def _ncdf(x: float) -> float:
    # only depend on scipy if absolutely necessary
    return (1.0 + erf(x * (2.0**-0.5))) * 0.5


def black_scholes(
    years_to_expiration: float,
    strike: float,
    spot: float,
    rate_of_return: float,
    volatility: float,
) -> float:
    rT = rate_of_return * years_to_expiration
    σT = volatility * np.sqrt(years_to_expiration)
    Z = np.exp(-rT)
    if np.isclose(σT, 0.0):
        return np.maximum(spot - strike * Z, 0.0)
    d1 = (np.log(spot / strike) + rT) / σT + σT * 0.5
    return spot * _ncdf(d1) - strike * Z * _ncdf(d1 - σT)
