import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №4")
print("Вариант 4: x^2 + y^2 = 4, xy = 1, (x0, y0) = (1.5, 0.7)")
print("=" * 60)

# =====================================================================
# ЗАДАНИЕ 1: Решение системы методом Ньютона
# =====================================================================

# Определяем систему уравнений F(x) = 0
# Вариант 4: x^2 + y^2 = 4, x*y = 1
# Переносим всё в левую часть:
# f1 = x^2 + y^2 - 4
# f2 = x*y - 1
def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        x[0] * x[1] - 1
    ])

def J(x):
    """
    Матрица Якоби для системы.
    df1/dx = 2x,  df1/dy = 2y
    df2/dx = y,   df2/dy = x
    """
    return np.array([
        [2*x[0], 2*x[1]],
        [x[1],   x[0]]
    ])

def newton_system(F, J, x0, eps=1e-8, max_iter=100):
    """
    Решает систему F(x)=0 методом Ньютона.
    x0 - начальное приближение (вектор).
    Возвращает (корень, число итераций, список приближений).
    """
    x = x0.astype(float).copy()
    history = [x.copy()]
    for i in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < eps:
            return x, i, history
        Jx = J(x)
        if abs(np.linalg.det(Jx)) < 1e-15:
            raise ValueError("Матрица Якоби вырождена")
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i + 1, history
        x = x_new
    raise RuntimeError("Метод Ньютона не сошёлся за заданное число итераций")

# Начальное приближение
x0 = np.array([1.5, 0.7])
root, iterations, history = newton_system(F, J, x0)
print("\n--- Задание 1: Решение системы методом Ньютона ---")
print(f"Начальное приближение: x0 = {x0}")
print(f"Найденный корень: x = {root}")
print(f"Число итераций: {iterations}")
print(f"Невязка ||F(x)||: {np.linalg.norm(F(root)):.2e}")

# =====================================================================
# ЗАДАНИЕ 2: Проверка точности
# =====================================================================

# Используем встроенный решатель scipy.optimize.fsolve (если установлен scipy)
try:
    from scipy.optimize import fsolve
    root_scipy = fsolve(F, x0)
    print("\n--- Задание 2: Проверка с помощью scipy.optimize.fsolve ---")
    print(f"Решение scipy: {root_scipy}")
    print(f"Невязка: {np.linalg.norm(F(root_scipy)):.2e}")
    print(f"Разница: {np.linalg.norm(root - root_scipy):.2e}")
except ImportError:
    print("\n--- Задание 2: scipy не установлен. Пропускаем. ---")

# =====================================================================
# ЗАДАНИЕ 3: Исследование сходимости
# =====================================================================

print("\n--- Задание 3: Исследование сходимости ---")
print(f"{'Нач. x':<10} {'Нач. y':<10} {'Итераций':<10} {'Корень x':<12} {'Корень y':<12} {'Невязка':<12}")
print("-" * 70)

starts = [
    (1.5, 0.7),    # начальное приближение из варианта
    (0.5, 1.5),
    (-1.5, -0.7),
    (-0.5, -1.5),
    (2.0, 1.0),
    (1.0, 1.0),    # на прямой x = y: det J = 0 (вырожденный якобиан)
    (3.0, -2.0),
]

for x0_, y0_ in starts:
    try:
        x0_vec = np.array([x0_, y0_])
        root_tmp, it_tmp, _ = newton_system(F, J, x0_vec)
        res_tmp = np.linalg.norm(F(root_tmp))
        print(f"{x0_:<10.2f} {y0_:<10.2f} {it_tmp:<10} {root_tmp[0]:<12.6f} {root_tmp[1]:<12.6f} {res_tmp:<12.2e}")
    except Exception as e:
        print(f"{x0_:<10.2f} {y0_:<10.2f} {'не сошёлся':<10} {'-':<12} {'-':<12} {'-':<12}")

# =====================================================================
# ЗАДАНИЕ 4: Визуализация
# =====================================================================

print("\n--- Задание 4: График сходимости ---")

# Строим линии уровня для системы
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = X**2 + Y**2 - 4
F2 = X * Y - 1

plt.figure(figsize=(8, 6))
c1 = plt.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2)
c2 = plt.contour(X, Y, F2, levels=[0], colors='green', linewidths=2)
plt.plot([], [], color='blue', linewidth=2, label='x^2+y^2=4')
plt.plot([], [], color='green', linewidth=2, label='xy=1')

# Отображаем итерации
history = np.array(history)
plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=1.5, markersize=4,
         label='Итерации Ньютона')
plt.plot(root[0], root[1], 'ks', markersize=8, label='Корень')

plt.title('Метод Ньютона для системы нелинейных уравнений')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('lab4_fig1.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)
