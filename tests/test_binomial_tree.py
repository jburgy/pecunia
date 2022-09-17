import pytest

from pecunia.atoms import At
from pecunia import binomial_tree, generate

testdata = [(At(1.0), pytest.approx(1.0))]


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
