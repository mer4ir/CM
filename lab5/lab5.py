import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №5")
print("Вариант 4: x = [0..5], y = x^2, k = 3")
print("=" * 60)

# =====================================================================
# TASK 1: Интерполяционный полином Лагранжа
# =====================================================================

def lagrange(x, x_nodes, y_nodes):
    """
    Вычисляет значение интерполяционного полинома Лагранжа в точке x.
    x_nodes, y_nodes - массивы узлов и значений.
    """
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return result

# Задаём узлы интерполяции (вариант 4)
x_nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_nodes = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])

# Строим интерполяционный полином на [x0 - 0.5, xn + 0.5] с шагом 0.05
x_plot = np.round(np.arange(x_nodes[0] - 0.5, x_nodes[-1] + 0.5 + 1e-9, 0.05), 2)
y_lagrange = np.array([lagrange(x, x_nodes, y_nodes) for x in x_plot])

print("\n--- Задание 1: Интерполяция Лагранжа ---")
print("Узлы: x =", x_nodes)
print("Значения: y =", y_nodes)
print(f"Число точек сетки: {len(x_plot)}")
print(f"{'x':<8} {'P(x)':<14} {'x^2':<14} {'|P(x)-x^2|':<12}")
print("-" * 50)
for x, p in zip(x_plot, y_lagrange):
    print(f"{x:<8.2f} {p:<14.8f} {x**2:<14.8f} {abs(p - x**2):<12.2e}")
print(f"Максимальное отклонение от x^2: {np.max(np.abs(y_lagrange - x_plot**2)):.2e}")

# =====================================================================
# TASK 2: Аппроксимация методом наименьших квадратов
# =====================================================================

def least_squares(x_nodes, y_nodes, degree):
    """
    Строит аппроксимирующий полином заданной степени методом наименьших квадратов.
    Возвращает коэффициенты полинома (от младшей степени к старшей).
    """
    # Система нормальных уравнений: A * coeff = b
    n = len(x_nodes)
    m = degree + 1
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(m):
        for j in range(m):
            A[i, j] = np.sum(x_nodes ** (i + j))
        b[i] = np.sum(y_nodes * (x_nodes ** i))
    coeff = np.linalg.solve(A, b)
    return coeff  # от младшей степени к старшей

degree = 3
coeff = least_squares(x_nodes, y_nodes, degree)

# Функция для вычисления аппроксимирующего полинома
def approx_poly(x, coeff):
    return np.polyval(coeff[::-1], x)  # np.polyval ожидает от старшей к младшей

y_approx = approx_poly(x_plot, coeff)

print("\n--- Задание 2: Аппроксимация МНК ---")
print(f"Степень полинома: {degree}")
print(f"Коэффициенты (от младшей к старшей): {coeff}")
print(f"Проверка np.polyfit (от старшей к младшей): {np.polyfit(x_nodes, y_nodes, degree)}")
S = np.sum((y_nodes - approx_poly(x_nodes, coeff)) ** 2)
print(f"Сумма квадратов отклонений S = {S:.3e}")

# =====================================================================
# TASK 3: Сравнение интерполяции и аппроксимации
# =====================================================================

print("\n--- Задание 3: Сравнение ---")
# Вычисляем отклонения в узлах
y_interp_nodes = [lagrange(x, x_nodes, y_nodes) for x in x_nodes]
y_approx_nodes = approx_poly(x_nodes, coeff)
print(f"{'Узел x':<8} {'y (исх.)':<10} {'Интерполяция':<15} {'Аппроксимация':<15} {'Ошибка интерп.':<16} {'Ошибка аппрокс.':<16}")
print("-" * 85)
for x, y_true, y_int, y_app in zip(x_nodes, y_nodes, y_interp_nodes, y_approx_nodes):
    print(f"{x:<8.2f} {y_true:<10.2f} {y_int:<15.6f} {y_app:<15.6f} {abs(y_true-y_int):<16.2e} {abs(y_true-y_app):<16.2e}")

# =====================================================================
# TASK 4: Визуализация
# =====================================================================

print("\n--- Задание 4: Графики ---")
plt.figure(figsize=(8, 5))
plt.plot(x_nodes, y_nodes, 'ko', markersize=8, label='Узлы')
plt.plot(x_plot, y_lagrange, 'r-', linewidth=2, label='Интерполяция Лагранжа')
plt.plot(x_plot, y_approx, 'b--', linewidth=2, label=f'Аппроксимация (степень {degree})')
plt.title('Интерполяция и аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('/Users/mera/Documents/tuition/CM/lab5/lab5_fig1.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# TASK 5: Исследование феномена Рунге
# =====================================================================

print("\n--- Задание 5: Феномен Рунге ---")
# Функция Рунге: f(x) = 1 / (1 + 25*x^2)
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# Интерполяция на равномерной сетке для разных n
n_values = [5, 8, 11, 15]
x_true = np.linspace(-1, 1, 300)
y_true = runge(x_true)

plt.figure(figsize=(8, 5))
plt.plot(x_true, y_true, 'k-', linewidth=2, label='Исходная функция')
print(f"{'n':<5} {'max|f-P| (равномерные)':<26} {'max|f-P| (Чебышёва)':<22}")
print("-" * 55)
errs_uniform, errs_cheb = [], []
for n in n_values:
    x_nodes_r = np.linspace(-1, 1, n)
    y_nodes_r = runge(x_nodes_r)
    y_interp_r = [lagrange(x, x_nodes_r, y_nodes_r) for x in x_true]
    plt.plot(x_true, y_interp_r, '--', label=f'Интерполяция n={n}')
    e_u = np.max(np.abs(np.array(y_interp_r) - y_true))
    # Для сравнения: узлы Чебышёва
    x_ch = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    y_ch = runge(x_ch)
    y_interp_c = [lagrange(x, x_ch, y_ch) for x in x_true]
    e_c = np.max(np.abs(np.array(y_interp_c) - y_true))
    errs_uniform.append(e_u)
    errs_cheb.append(e_c)
    print(f"{n:<5} {e_u:<26.4e} {e_c:<22.4e}")

plt.title('Феномен Рунге: интерполяция функции 1/(1+25x^2)')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.legend()
plt.grid(True)
plt.savefig('/Users/mera/Documents/tuition/CM/lab5/lab5_fig2.png', dpi=150, bbox_inches='tight')
plt.show()

# Способ борьбы: узлы Чебышёва вместо равномерных
plt.figure(figsize=(8, 5))
plt.plot(x_true, y_true, 'k-', linewidth=2, label='Исходная функция')
for n in n_values:
    x_ch = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    y_ch = runge(x_ch)
    y_interp_c = [lagrange(x, x_ch, y_ch) for x in x_true]
    plt.plot(x_true, y_interp_c, '--', label=f'Узлы Чебышёва n={n}')
plt.title('Интерполяция на узлах Чебышёва')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('/Users/mera/Documents/tuition/CM/lab5/lab5_fig3.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)
