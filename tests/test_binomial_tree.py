import numpy as np
import pytest

from pecunia import binomial_tree, generate
from pecunia.atoms import And, At, Or

from .utils import black_scholes

testdata = [
    # zero coupon bond
    (At(1.0), 0.2, pytest.approx(1.0)),
]

for volatility in np.arange(0.0, 0.5, 0.1):
    testdata.append(
        # european call option
        (
            Or([And([At(1.0), -1.0]), 0.0]),
            volatility,
            pytest.approx(
                black_scholes(
                    years_to_expiration=1.0,
                    strike=1.0,
                    spot=1.0,
                    rate_of_return=0.02,
                    volatility=max(volatility, volatility),
                ),
                rel=1e-3,
            ),
        ),
    )


@pytest.mark.parametrize("implementation", ["ast", "bytecode"])
@pytest.mark.parametrize("contract,volatility,expected", testdata)
def test_binomial_tree(implementation, contract, volatility, expected):
    evolution = generate.from_graph(contract, implementation=implementation)
    present_value = binomial_tree.present_value(
        evolution,
        years_to_expiration=1.0,
        spot=1.0,
        rate_of_return=0.02,
        volatility=volatility,
    )
    assert present_value == expected
