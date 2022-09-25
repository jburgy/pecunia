import pytest

from pecunia import binomial_tree, generate
from pecunia.atoms import And, At, Or

from .utils import black_scholes

testdata = [
    # zero coupon bond
    (At(1.0), pytest.approx(1.0)),
    # european call option
    (
        Or([And([At(1.0), -1.0]), 0.0]),
        pytest.approx(
            black_scholes(
                years_to_expiration=1.0,
                strike=1.0,
                spot=1.0,
                rate_of_return=0.02,
                volatility=0.3,
            ),
            abs=1e-4,
        ),
    ),
]


@pytest.mark.parametrize("contract,expected", testdata)
def test_binomial_tree(contract, expected):
    evolution = generate.from_graph(contract)
    present_value = binomial_tree.present_value(
        evolution,
        years_to_expiration=1.0,
        spot=1.0,
        rate_of_return=0.02,
        volatility=0.3,
    )
    assert present_value == expected
