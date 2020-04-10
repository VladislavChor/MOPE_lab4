from copy import deepcopy
from math import sqrt
import numpy as np
from prettytable import PrettyTable

x1_min = -20
x1_max = 15
x2_min = -35
x2_max = 10
x3_min = 10
x3_max = 20

x_average_max = (x1_max + x2_max + x3_max) / 3
x_average_min = (x1_min + x2_min + x3_min) / 3
y_max = 200 + x_average_max
y_min = 200 + x_average_min


def replace_column(list_: list, column, list_replace):
    list_ = deepcopy(list_)
    for i in range(len(list_)):
        list_[i][column] = list_replace[i]
    return list_


def main(m, n):
    if n == 8:
        print(
            'ŷ = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b12 * x1 * x2 + b13 * x1 * x3 + b23 * x2 * x3 + b123 * x1 * x2 * x3')
        norm_x = [
            [+1, -1, -1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
            [+1, -1, -1, +1],
            [+1, -1, +1, -1],
            [+1, +1, -1, -1],
            [+1, +1, +1, +1]
        ]

        for i in range(len(norm_x)):
            norm_x[i].append(norm_x[i][1] * norm_x[i][2])
            norm_x[i].append(norm_x[i][1] * norm_x[i][3])
            norm_x[i].append(norm_x[i][2] * norm_x[i][3])
            norm_x[i].append(norm_x[i][1] * norm_x[i][2] * norm_x[i][3])

        x = [
            [x1_min, x2_min, x3_min],
            [x1_min, x2_max, x3_max],
            [x1_max, x2_min, x3_max],
            [x1_max, x2_max, x3_min],
            [x1_min, x2_min, x3_max],
            [x1_min, x2_max, x3_min],
            [x1_max, x2_min, x3_min],
            [x1_max, x2_max, x3_max]
        ]
        for i in range(len(x)):
            x[i].append(x[i][0] * x[i][1])
            x[i].append(x[i][0] * x[i][2])
            x[i].append(x[i][1] * x[i][2])
            x[i].append(x[i][0] * x[i][1] * x[i][2])

    if n == 4:
        print('ŷ = b0 + b1 * x1 + b2 * x2 + b3 * x3')
        norm_x = [
            [+1, -1, -1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
        ]
        x = [
            [x1_min, x2_min, x3_min],
            [x1_min, x2_max, x3_max],
            [x1_max, x2_min, x3_max],
            [x1_max, x2_max, x3_min],
        ]
    y = np.random.randint(y_min, y_max, size=(n, m))
    y_av = list(np.average(y, axis=1))

    for i in range(len(y_av)):
        y_av[i] = round(y_av[i], 3)
    if n == 8:
        t = PrettyTable(['N', 'norm_x_0', 'norm_x_1', 'norm_x_2', 'norm_x_3', 'norm_x_1_x_2', 'norm_x_1_x_3',
                         'norm_x_2_x_3', 'norm_x_1_x_2_x_3', 'x_1', 'x_2', 'x_3', 'x_1_x_2', 'x_1_x_3', 'x_2_x_3',
                         'x_1_x_2_x_3'] + [f'y_{i + 1}' for i in range(m)] + ['y_av'])
        for i in range(n):
            t.add_row([i + 1] + list(norm_x[i]) + list(x[i]) + list(y[i]) + [y_av[i]])
        print(t)
        sums_of_columns_x = np.sum(x, axis=0)
        m_ij = [[n] + [i for i in sums_of_columns_x]]
        for i in range(len(sums_of_columns_x)):
            m_ij.append(
                [sums_of_columns_x[i]] + [sum([x[k][i] * x[k][j] for k in range(len(x[i]))]) for j in range(len(x[i]))])

        k_i = [sum(y_av)]
        for i in range(len(sums_of_columns_x)):
            k_i.append(sum(y_av[j] * x[j][i] for j in range(len(x[i]))))

        det = np.linalg.det(m_ij)
        det_i = [np.linalg.det(replace_column(m_ij, i, k_i)) for i in range(len(k_i))]

        b_i = [i / det for i in det_i]

        print(
            f"\nНормалізоване рівняння регресії: y = {b_i[0]:.5f} + {b_i[1]:.5f} * x1 + {b_i[2]:.5f} * x2 + "
            f"{b_i[3]:.5f} * x3 + {b_i[4]:.5f} * x1 * x2 + "
            f"{b_i[5]:.5f} * x1 * x3 + {b_i[6]:.5f} * x2 * x3 + {b_i[7]:.5f} * x1 * x2 * x3")

    if n == 4:
        t = PrettyTable(
            ['N', 'norm_x_0', 'norm_x_1', 'norm_x_2', 'norm_x_3', 'x_1', 'x_2', 'x_3'] + [f'y_{i + 1}' for i in
                                                                                          range(m)] + ['y_av'])
        for i in range(n):
            t.add_row([i + 1] + list(norm_x[i]) + list(x[i]) + list(y[i]) + [y_av[i]])
        print(t)

        mx_1, mx_2, mx_3 = [i / len(x) for i in np.sum(x, axis=0)]
        my = sum(y_av) / len(y_av)

        a_1 = sum([x[i][0] * y_av[i] for i in range(len(x))]) / len(x)
        a_2 = sum([x[i][1] * y_av[i] for i in range(len(x))]) / len(x)
        a_3 = sum([x[i][2] * y_av[i] for i in range(len(x))]) / len(x)

        a_11 = sum([x[i][0] ** 2 for i in range(len(x))]) / len(x)
        a_22 = sum([x[i][1] ** 2 for i in range(len(x))]) / len(x)
        a_33 = sum([x[i][2] ** 2 for i in range(len(x))]) / len(x)
        a_12 = sum([x[i][0] * x[i][1] for i in range(len(x))]) / len(x)
        a_13 = sum([x[i][0] * x[i][2] for i in range(len(x))]) / len(x)
        a_23 = a_32 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / len(x)

        matrix = [
            [1, mx_1, mx_2, mx_3],
            [mx_1, a_11, a_12, a_13],
            [mx_2, a_12, a_22, a_32],
            [mx_3, a_13, a_23, a_33]
        ]

        answers = [my, a_1, a_2, a_3]

        det = np.linalg.det(matrix)
        det_i = [np.linalg.det(replace_column(matrix, i, answers)) for i in range(len(answers))]

        b_i = [i / det for i in det_i]
        print(
            f"\nНормалізоване рівняння регресії: y = {b_i[0]:.5f} + {b_i[1]:.5f} * x1 + {b_i[2]:.5f} * x2 + {b_i[3]:.5f} * x3\n")

    print("\nКритерій Кохрена")
    f_1 = m - 1
    f_2 = n
    s_i = [sum([(i - y_av[j]) ** 2 for i in y[j]]) / m for j in range(len(y))]
    g_p = max(s_i) / sum(s_i)
    # Розподіл Кохрена
    table = {3: 0.6841, 4: 0.6287, 5: 0.5892, 6: 0.5598, 7: 0.5365, 8: 0.5175, 9: 0.5017, 10: 0.4884,
             range(11, 17): 0.4366, range(17, 37): 0.3720, range(37, 145): 0.3093}
    g_t = table.get(m)

    if g_p < g_t:
        print(f"Дисперсія однорідна: Gp = {g_p:.5} < Gt = {g_t}")
    else:
        print(f"Дисперсія неоднорідна: Gp = {g_p:.5} < Gt = {g_t}\nЗбільшуємо m на 1")
        return main(m=m + 1, n=n)

    print("\nКритерій Стьюдента")
    s2_b = sum(s_i) / n
    s2_beta_s = s2_b / (n * m)
    s_beta_s = sqrt(s2_beta_s)

    beta_i = [sum([norm_x[i][j] * y_av[i] for i in range(len(norm_x))]) / n for j in range(len(norm_x))]

    t = [abs(i) / s_beta_s for i in beta_i]

    f_3 = f_1 * f_2
    #  Розподіл Стьюдента
    t_table = {8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
               17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.08, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.06}
    d = deepcopy(n)
    for i in range(len(t)):
        if t_table.get(f_3) > t[i]:
            beta_i[i] = 0
            d -= 1
    if n == 8:
        print(
            f"\nРівняння регресії: y = {beta_i[0]:.5f} + {beta_i[1]:.5f} * x1 + {beta_i[2]:.5f} * x2 + "
            f"{beta_i[3]:.5f} * x3 + {beta_i[4]:.5f} * x1 * x2 + "
            f"{beta_i[5]:.5f} * x1 * x3 + {beta_i[6]:.5f} * x2 * x3 + {beta_i[7]:.5f} * x1 * x2 * x3")
        check_i = [
            beta_i[0] + beta_i[1] * i[0] + beta_i[2] * i[1] + beta_i[3] * i[2] + beta_i[4] * i[3] + beta_i[5] * i[4] +
            beta_i[6] * i[5] + beta_i[7] * i[6] for i in x]
        print("Значення нормалізовані: ", check_i)

    if n == 4:
        print(
            f"\nРівняння регресії: y = {beta_i[0]:.5f} + {beta_i[1]:.5f} * x1 + {beta_i[2]:.5f} * x2 + "
            f"{beta_i[3]:.5f} * x3")
        check_i = [
            beta_i[0] + beta_i[1] * i[0] + beta_i[2] * i[1] + beta_i[3] * i[2] for i in x]
        print("Значення нормалізовані: ", check_i)


    print("\nКритерій Фішера")
    f_4 = n - d
    s2_ad = m / f_4 * sum([(check_i[i] - y_av[i]) ** 2 for i in range(len(y_av))])
    f_p = s2_ad / s2_b
    # Розподіл Фішера
    f_t = [
        [164.4, 199.5, 215.7, 224.6, 230.2, 234, 235.8, 237.6],
        [18.5, 19.2, 19.2, 19.3, 19.3, 19.3, 19.4, 19.4],
        [10.1, 9.6, 9.3, 9.1, 9, 8.9, 8.8, 8.8],
        [7.7, 6.9, 6.6, 6.4, 6.3, 6.2, 6.1, 6.1],
        [6.6, 5.8, 5.4, 5.2, 5.1, 5, 4.9, 4.9],
        [6, 5.1, 4.8, 4.5, 4.4, 4.3, 4.2, 4.2],
        [5.5, 4.7, 4.4, 4.1, 4, 3.9, 3.8, 3.8],
        [5.3, 4.5, 4.1, 3.8, 3.7, 3.6, 3.5, 3.5],
        [5.1, 4.3, 3.9, 3.6, 3.5, 3.4, 3.3, 3.3],
        [5, 4.1, 3.7, 3.5, 3.3, 3.2, 3.1, 3.1],
        [4.8, 4, 3.6, 3.4, 3.2, 3.1, 3, 3],
        [4.8, 3.9, 3.5, 3.3, 3.1, 3, 2.9, 2.9],
        [4.7, 3.8, 3.4, 3.2, 3, 2.9, 2.8, 2.8],
        [4.6, 3.7, 3.3, 3.1, 3, 2.9, 2.8, 2.7],
        [4.5, 3.7, 3.3, 3.1, 2.9, 2.8, 2.7, 2.7],
        [4.5, 3.6, 3.2, 3, 2.9, 2.7, 2.6, 2.6],
        [4.5, 3.6, 3.2, 3, 2.8, 2.7, 2.5, 2.3],
        [4.4, 3.6, 3.2, 2.9, 2.8, 2.7, 2.5, 2.3],
        [4.4, 3.5, 3.1, 2.9, 2.7, 2.7, 2.4, 2.3],
        [4.4, 3.5, 3.1, 2.8, 2.7, 2.7, 2.4, 2.3],
        [4.4, 3.5, 3.1, 2.8, 2.7, 2.6, 2.4, 2.3],
        [4.3, 3.4, 3.1, 2.8, 2.7, 2.6, 2.4, 2.3],
        [4.3, 3.4, 3.1, 2.8, 2.6, 2.6, 2.3, 2.2],
        [4.3, 3.4, 3, 2.8, 2.6, 2.5, 2.3, 2.2],
        [4.3, 3.4, 3, 2.8, 2.6, 2.5, 2.3, 2.2],
    ]
    if f_p > f_t[f_3][f_4]:
        print(
            f"fp = {f_p} > ft = {f_t[f_3][f_4]}.\nМатематична модель неадекватна експериментальним  "
            f"даним\nЗбільшуємо m на 1")
        main(m=m + 1, n=8)
    else:
        print(f"fP = {f_p} < fT = {f_t[f_3][f_4]}.\nМатематична модель адекватна експериментальним даним\n")


main(m=3, n=4)