import pyomo.environ as pyo
import numpy as np
from timeit import repeat
from scipy.sparse import coo_matrix

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptNLP, CyIpoptSolver
from suspect.pyomo import read_osil
from cppyad.ipopt import CppADIpoptNLP

#m = read_osil("/home/fra/Downloads/minlplib/osil/faclay20h.osil", objective_prefix="obj_")
m = read_osil("/home/fra/Downloads/minlplib/osil/alan.osil", objective_prefix="obj_")

for var in m.component_data_objects(pyo.Var, active=True, descend_into=True):
    if var.lb is None:
        continue
    if var.ub is None:
        continue
    if np.isclose(var.lb, var.ub):
        var.setlb(var.lb - 1e-4)
    if var.domain != pyo.Reals:
        var.domain = pyo.Reals

x = [
    1.0
    for _ in m.component_data_objects(pyo.Var, active=True, descend_into=True)
]


cad = CppADIpoptNLP(m)
nlp = CyIpoptNLP(PyomoNLP(m))

for f in ['x_init', 'x_lb', 'x_ub', 'g_lb', 'g_ub']:
    cad_func = getattr(cad, f)
    cad_r = cad_func()
    nlp_func = getattr(nlp, f)
    nlp_r = nlp_func()
    eq = False
    try:
        eq = np.all(np.isclose(cad_r, nlp_r))
    except:
        pass
    print(f, '   ', eq)
    print('cad ', cad_r)
    print('nlp ', nlp_r)
    print()

for f in ['objective', 'gradient', 'constraints']:
    cad_func = getattr(cad, f)
    cad_r = cad_func(x)
    nlp_func = getattr(nlp, f)
    nlp_r = nlp_func(x)
    eq = False
    try:
        eq = np.all(np.isclose(cad_r, nlp_r))
    except:
        pass
    print(f, '   ', eq)
    print('cad ', cad_r)
    print('nlp ', nlp_r)
    print()


cad_jac = coo_matrix((cad.jacobian(x), cad.jacobianstructure()))
nlp_jac = coo_matrix((nlp.jacobian(x), nlp.jacobianstructure()))
print('Jacobian')
print('cad')
print(cad_jac.toarray())
print('nlp')
print(nlp_jac.toarray())

if False:
    nlp_solver = CyIpoptSolver(nlp, {'derivative_test': 'first-order'})
    nlp_solver.solve(tee=True)

    cad_solver = CyIpoptSolver(cad, {'derivative_test': 'first-order'})
    cad_solver.solve(tee=True)