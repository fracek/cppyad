import numpy as np
import pyomo.environ as pe
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptProblemInterface
from pyomo.core.expr.visitor import ExpressionValueVisitor, identify_variables
from pyomo.core.expr.current import nonpyomo_leaf_types
from cppyad_core import independent, ADFun_double, SparseJacobianWork, VectorArrayWrapper


def build_adfun(model, active=True):
    variables = list(model.component_data_objects(pe.Var, active=active, descend_into=True))
    ad_variables = independent([0.0] * len(variables))
    var_to_ad_map = pe.ComponentMap()
    var_to_idx_map = pe.ComponentMap()
    variables_count = 0
    x_init = []
    x_lb = []
    x_ub = []
    for i, v in enumerate(variables):
        var_to_ad_map[v] = ad_variables[i]
        v_lb = v.lb
        v_ub = v.ub
        if v_lb is None:
            v_lb = -np.inf
        if v_ub is None:
            v_ub = np.inf
        v_i = pe.value(v, exception=False)
        if v_i is None:
            if np.isinf(v_lb) and np.isinf(v_ub):
                v_i = 0.0
            elif np.isinf(v_lb):
                v_i = v_ub
            else:
                v_i = v_lb
        x_init.append(v_i)
        x_lb.append(v_lb)
        x_ub.append(v_ub)
        var_to_idx_map[v] = i
        variables_count += 1

    visitor = ADFunBuilderVisitor(var_to_ad_map)

    ad_y = []
    objective_count = 0
    for obj in model.component_data_objects(pe.Objective, active=active, descend_into=True):
        ad_y.append(visitor.dfs_postorder_stack(obj.expr))
        objective_count += 1

    constraint_count = 0
    g_lb = []
    g_ub = []
    for con in model.component_data_objects(pe.Constraint, active=active, descend_into=True):
        ad_y.append(visitor.dfs_postorder_stack(con.body))
        constraint_count += 1
        if con.lower is None:
            g_lb.append(-np.inf)
        else:
            g_lb.append(pe.value(con.lower))

        if con.upper is None:
            g_ub.append(np.inf)
        else:
            g_ub.append(pe.value(con.upper))

    f = ADFun_double(ad_variables, ad_y)

    x_init = np.array(x_init, dtype=np.float64)
    x_lb = np.array(x_lb, dtype=np.float64)
    x_ub = np.array(x_ub, dtype=np.float64)

    g_lb = np.array(g_lb, dtype=np.float64)
    g_ub = np.array(g_ub, dtype=np.float64)
    return f, variables_count, objective_count, constraint_count, x_init, x_lb, x_ub, g_lb, g_ub
    if False:
        row_len = objective_count + constraint_count
        col_len = variables_count
        pattern_jac = np.zeros(col_len * row_len, dtype=np.bool)
        row_count = 0
        el_count = 0
        for obj in model.component_data_objects(pe.Objective, active=active, descend_into=True):
            variables = identify_variables(obj.expr, include_fixed=True)
            for v in variables:
                pattern_jac[row_count * col_len + var_to_idx_map[v]] = 1.0
                el_count += 1
            row_count += 1

        for con in model.component_data_objects(pe.Constraint, active=active, descend_into=True):
            variables = identify_variables(con.body, include_fixed=True)
            for v in variables:
                pattern_jac[row_count * col_len + var_to_idx_map[v]] = 1.0
                el_count += 1
            row_count += 1

        row_jac = np.zeros(el_count, dtype=np.int)
        col_jac = np.zeros(el_count, dtype=np.int)
        el_idx = 0
        for i in range(constraint_count + objective_count):
            for j in range(variables_count):
                if pattern_jac[i * col_len + j] == 1:
                    row_jac[el_idx] = i
                    col_jac[el_idx] = j
                    el_idx += 1
        assert el_idx == el_count

        f = ADFun_double(ad_variables, ad_y)

        r = np.eye(col_len).reshape(col_len*col_len)
        s = f.sparse_jacobian_pattern(col_len, r)
        print(s)
        return f, pattern_jac, row_jac, col_jac, row_len, col_len


class ADFunBuilderVisitor(ExpressionValueVisitor):
    def __init__(self, var_map):
        super().__init__()
        self._var_map = var_map

    def visit(self, node, values):
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_variable_type():
            return True, self._var_map[node]

        if not node.is_expression_type():
            raise ValueError("??", node)

        return False, None


def _jacobian_structure(adfun, nx, nf, ng):
    m = nf + ng
    r = np.eye(nx).reshape(nx*nx)
    jac_pat = adfun.sparse_jacobian_pattern(nx, r)
    jac_row = []
    jac_col = []
    jac_row_ipopt = []
    jac_col_ipopt = []
    grad_col_ipopt = []
    jac_skip_ipopt = None
    skip_count = 0
    for i in range(m):
        for j in range(nx):
            if jac_pat[i * nx + j]:
                skip_count += 1
                jac_row.append(i)
                jac_col.append(j)
                if i >= nf:
                    if jac_skip_ipopt is None:
                        jac_skip_ipopt = skip_count
                    jac_row_ipopt.append(i - nf)
                    jac_col_ipopt.append(j)
                else:
                    grad_col_ipopt.append(j)

    return jac_pat, jac_row, jac_col, jac_row_ipopt, jac_col_ipopt, jac_skip_ipopt, grad_col_ipopt


class CppADIpoptNLP(CyIpoptProblemInterface):
    def __init__(self, model, active=True, use_sparse_gradient=True):
        self.use_sparse_gradient = use_sparse_gradient

        adfun, nx, nf, ng, x_init, x_lb, x_ub, g_lb, g_ub = build_adfun(model, active)
        self._adfun = adfun
        self._nx = nx
        self._nf = nf
        self._ng = ng

        self._x_init = x_init
        self._x_lb = x_lb
        self._x_ub = x_ub

        self._g_lb = g_lb
        self._g_ub = g_ub

        self._cached_x = x_init.copy()

        self._fg0 = adfun.forward(0, self._x_init)
        self._w = np.zeros(self._nf + self._ng, dtype=np.float64)
        self._grad = np.zeros(self._nx, dtype=np.float64)

        jac_pat, jac_row, jac_col, jac_row_ipopt, jac_col_ipopt, jac_skip_ipopt, grad_col_ipopt = _jacobian_structure(adfun, nx, nf, ng)
        self._jac_pat = jac_pat
        self._jac_row = jac_row
        self._jac_col = jac_col
        self._jac_row_ipopt = jac_row_ipopt
        self._jac_col_ipopt = jac_col_ipopt
        self._jac_skip_ipopt = jac_skip_ipopt
        self._jac_work = SparseJacobianWork()
        self._jac_vec = VectorArrayWrapper(len(jac_row))
        self._grad_col_ipopt = grad_col_ipopt

        self._fg0_cached = False
        self._jac_cached = False

    def _invalidate_caches(self):
        self._fg0_cached = False
        self._jac_cached = False

    def _cache_new_x(self, x):
        if not np.array_equal(x, self._cached_x):
            self._invalidate_caches()
            np.copyto(self._cached_x, x)

    def _compute_fg0(self):
        if self._fg0_cached:
            return
        self._fg0 = self._adfun.forward(0, self._cached_x)
        self._fg0_cached = True

    def _compute_jacobian(self):
        if self._jac_cached:
            return
        self._adfun.sparse_jacobian_forward(
            self._cached_x, self._jac_pat, self._jac_row, self._jac_col, self._jac_vec, self._jac_work
        )

    def x_init(self):
        return self._x_init

    def x_lb(self):
        return self._x_lb

    def x_ub(self):
        return self._x_ub

    def g_lb(self):
        return self._g_lb

    def g_ub(self):
        return self._g_ub

    def objective(self, x):
        self._cache_new_x(x)
        self._compute_fg0()
        sum = 0.0
        for i in range(self._nf):
            sum += self._fg0[i]
        return sum

    def gradient(self, x):
        self._cache_new_x(x)
        if not self.use_sparse_gradient:
            nf = self._nf
            ng = self._ng
            for i in range(nf):
                self._w[i] = 1.0
            for i in range(self._ng):
                self._w[nf + i] = 0.0
            return self._adfun.reverse(1, self._w)
        self._compute_jacobian()
        jac = self._jac_vec.as_pyarray()
        for i, j in enumerate(self._grad_col_ipopt):
            self._grad[j] = jac[i]
        return self._grad

    def constraints(self, x):
        self._cache_new_x(x)
        self._compute_fg0()
        return self._fg0[self._nf:]

    def jacobianstructure(self):
        return self._jac_row_ipopt, self._jac_col_ipopt

    def jacobian(self, x):
        self._cache_new_x(x)
        self._compute_jacobian()
        return self._jac_vec.as_pyarray()[self._jac_skip_ipopt-1:]

    def hessianstructure(self):
        pass

    def hessian(self, x, y, obj_factor):
        pass
