# Copyright 2020 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pyomo.environ as pe
from cppyad._core import independent, ADFun
from pyomo.core.expr.current import nonpyomo_leaf_types
from pyomo.core.expr.visitor import ExpressionValueVisitor


def build_adfun_from_model(model, active=True):
    variables = list(
        model.component_data_objects(pe.Var, active=active, descend_into=True)
    )
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
    for obj in model.component_data_objects(pe.Objective,
                                            active=active,
                                            descend_into=True):
        ad_y.append(visitor.dfs_postorder_stack(obj.expr))
        objective_count += 1

    constraint_count = 0
    g_lb = []
    g_ub = []
    for con in model.component_data_objects(pe.Constraint,
                                            active=active,
                                            descend_into=True):
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

    f = ADFun(ad_variables, ad_y)

    x_init = np.array(x_init, dtype=np.float64)
    x_lb = np.array(x_lb, dtype=np.float64)
    x_ub = np.array(x_ub, dtype=np.float64)

    g_lb = np.array(g_lb, dtype=np.float64)
    g_ub = np.array(g_ub, dtype=np.float64)
    return (
        f,
        variables_count,
        objective_count,
        constraint_count,
        x_init,
        x_lb,
        x_ub,
        g_lb,
        g_ub,
    )


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
