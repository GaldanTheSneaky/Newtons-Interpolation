import numpy as np


def fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f


def uniform_divided_diff(y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1])
    return coef


def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef



def eval_pol(diff_table, x_data, x, der_order=0):
    n = len(diff_table)

    if der_order == 0:
        n = len(diff_table)
        sum = diff_table[0]

        for i in range(1, n):
            x_product = 1
            for j in range(i):
                x_product *= x - x_data[j]
            x_product *= diff_table[i]
            sum += x_product

    elif der_order == 1:
        sum = diff_table[1]

        for i in range(2, n):
            x_sum = 0
            for j in range(i):
                x_product = 1
                for k in range(i):
                    if k != j:
                        x_product *= x - x_data[k]
                x_sum += x_product

            x_sum *= diff_table[i]
            sum += x_sum

    elif der_order == 2:
        sum = diff_table[2]

        for i in range(3, n):
            x_sum = 0
            for j in range(i):
                x_sum2 = 0
                for k in range(i):
                    x_product = 1
                    if k != j:
                        for l in range(i):
                            if l != k and l != j:
                                x_product *= x - x_data[l]
                        x_sum += x_product
                x_sum += x_sum2

            x_sum *= diff_table[i]
            sum += x_sum
    else:
        raise Exception("Invalid order of derivative")

    return sum


def uniform_eval_pol(lower_bound, h, diff_table, n, x, der_order=0):

    q = (x - lower_bound) / h
    n = len(diff_table)

    if der_order == 0:
        sum = diff_table[0]

        for i in range(1, n):
            q_product = 1
            for j in range(i):
                q_product *= q - j
            q_product *= diff_table[i] / fact(i)
            sum += q_product

    elif der_order == 1:
        sum = diff_table[1]

        for i in range(2, n):
            q_sum = 0
            for j in range(i):
                q_product = 1
                for k in range(i):
                    if k != j:
                        q_product *= q - k
                q_sum += q_product

            q_sum *= diff_table[i] / fact(i)
            sum += q_sum
            sum *= 1/h

    elif der_order == 2:
        sum = diff_table[2]

        for i in range(3, n):
            q_sum = 0
            for j in range(i):
                q_sum2 = 0
                for k in range(i):
                    q_product = 1
                    if k != j:
                        for l in range(i):
                            if l != k and l != j:
                                q_product *= q - l
                        q_sum += q_product
                q_sum += q_sum2

            q_sum *= diff_table[i] / fact(i)
            sum += q_sum
            sum *= 1/(h * h)
    else:
        raise Exception("Invalid order of derivative")

    return sum