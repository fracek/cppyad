# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from cppyad import build_adfun_from_model


@pytest.fixture()
def model():
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0, 10), initialize=0.0)
    m.y = pe.Var(bounds=(-10, 10), initialize=0.0)

    m.obj = pe.Objective(expr=m.x + m.y)
    m.c0 = pe.Constraint(expr=m.x * m.y >= 0)
    return m


def test_build_adfun(model):
    adfun, nx, nf, ng, x_init, x_lb, x_ub, g_lb, g_ub = \
        build_adfun_from_model(model, active=True)
    assert nx == 2
    assert nf == 1
    assert ng == 1
    assert np.allclose(x_init, [.0, .0])
    assert np.allclose(x_lb, [0.0, -10.0])
    assert np.allclose(x_ub, [10.0, 10.0])
    assert np.allclose(g_lb, [0.0])
