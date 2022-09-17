from dis import dis

from pecunia.atoms import At
from pecunia import binomial_tree, generate

evolution = generate.from_graph(At(1.0))
dis(evolution)

present_value = binomial_tree.present_value(
    evolution,
    years_to_expiration=1.0,
    spot=1.0,
    rate_of_return=0.02,
    volatility=0.3,
)
