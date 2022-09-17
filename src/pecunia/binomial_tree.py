import numpy as np

from .generate import Evolution


def present_value(
    evolve: Evolution,
    years_to_expiration: float,
    spot: float,
    rate_of_return: float,
    volatility: float,
    steps: int = 1_000,
):
    dt = years_to_expiration / steps
    z = np.exp(-rate_of_return * dt)
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    p = (1 / z - d) / (u - d)
    q = 1 - p

    x = spot * np.geomspace(d**steps, u**steps, steps + 1)
    g = evolve(years_to_expiration, x, x * 0.0)
    v = g.send(None)

    times = np.linspace(years_to_expiration, 0, steps + 1)
    for time in times[1:]:
        x = u * x[:-1]
        v = z * (p * v[1:] + q * v[:-1])
        v = g.send((time, x, v))

    return v[0]
