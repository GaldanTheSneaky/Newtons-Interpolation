import numpy as np
import newtonsInterpolation as ni
import sympy as sym
from sklearn.metrics import mean_squared_error


def str_to_bool(v):
  return v.lower() in ("true\n", "1\n")


file = open('input.txt', 'r')

order_of_derivative = int(file.readline())
order_of_polynomial = int(file.readline())
is_uniform = str_to_bool(file.readline())
x_data = file.readline()
y_data = np.array([float(n) for n in file.readline().split(' ')])
number_of_intervals = int(file.readline())
x_new = np.array([float(n) for n in file.readline().split(' ')])
is_func = str_to_bool(file.readline())
function = file.readline()
file.close()

if is_uniform:
    lower_bound, upper_bound = [float(n) for n in x_data.split(' ')]
    h = (upper_bound - lower_bound) / (len(y_data) - 1)
    diff_table = ni.uniform_divided_diff(y_data)[0, :]
    y_new = ni.uniform_eval_pol(lower_bound, h, diff_table, len(y_data) - 1, x_new, order_of_derivative)
else:
    x_data = np.array([int(n) for n in x_data.split(' ')])
    diff_table = ni.divided_diff(x_data, y_data)[0, :]
    y_new = ni.eval_pol(diff_table, x_data, x_new, order_of_derivative)

file = open('output.txt', 'w')

file.write(str(y_new))
file.write('\n')
if is_func:
    x = sym.Symbol('x')
    func = eval(function)
    func_der = func.diff(x)
    func_second_der = func_der.diff(x)
    if order_of_derivative == 0:
        orig_function = sym.lambdify(x, func)
    elif order_of_derivative == 1:
        orig_function = sym.lambdify(x, func_der)
    elif order_of_derivative == 2:
        orig_function = sym.lambdify(x, func_second_der)

    file.write(str(mean_squared_error(y_new, orig_function(x_new))))
