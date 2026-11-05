import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("LABORATORY WORK #6")
print("Variant 4: f(x) = ln x, x0 = 2.0, [a, b] = [1, 3]")
print("=" * 60)

# =====================================================================
# TASK 1: Numerical differentiation
# =====================================================================

def f(x):
    return np.log(x)

def df_analytical(x):
    return 1.0 / x

def left_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def right_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 2.0
h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

print("\n--- TASK 1: Numerical differentiation ---")
print(f"x0 = {x0}, exact f'(x0) = {df_analytical(x0):.8f}")
print(f"{'h':<10} {'Left':<15} {'Right':<15} {'Central':<15} {'Left err':<12} {'Right err':<12} {'Central error':<15}")
print("-" * 100)

errors_left, errors_right, errors_central = [], [], []
for h in h_values:
    dl = left_diff(f, x0, h)
    dr = right_diff(f, x0, h)
    dc = central_diff(f, x0, h)
    err_l = abs(dl - df_analytical(x0))
    err_r = abs(dr - df_analytical(x0))
    err_c = abs(dc - df_analytical(x0))
    errors_left.append(err_l)
    errors_right.append(err_r)
    errors_central.append(err_c)
    print(f"{h:<10.0e} {dl:<15.8f} {dr:<15.8f} {dc:<15.8f} {err_l:<12.2e} {err_r:<12.2e} {err_c:<15.2e}")

# Оптимальный шаг - тот, при котором ошибка минимальна
for name, errs in [("Left", errors_left), ("Right", errors_right), ("Central", errors_central)]:
    k = int(np.argmin(errs))
    print(f"Optimal step ({name}): h = {h_values[k]:.0e}, error = {errs[k]:.2e}")

plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors_left, 'o-', label='Left difference')
plt.loglog(h_values, errors_right, 's-', label='Right difference')
plt.loglog(h_values, errors_central, '^-', label='Central difference')
plt.title('Differentiation errors vs h')
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('/Users/mera/Documents/tuition/CM/lab6/lab6_fig1.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# TASK 2: Error investigation
# =====================================================================

print("\n--- TASK 2: Error vs h for central difference ---")
# Порядок точности: наклон прямой в log-log масштабе на участке 1e-1 ... 1e-4
hh = np.array(h_values[:4])
ee = np.array(errors_central[:4])
order = np.polyfit(np.log10(hh), np.log10(ee), 1)[0]
print(f"Estimated order of accuracy (h = 1e-1 ... 1e-4): p = {order:.3f}")
for i in range(3):
    p = np.log10(errors_central[i] / errors_central[i + 1])
    print(f"  h {h_values[i]:.0e} -> {h_values[i+1]:.0e}: p = {p:.3f}")

plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors_central, 'o-', color='red', linewidth=2, label='Central difference error')
plt.loglog(h_values, [errors_central[0] * (h / h_values[0]) ** 2 for h in h_values],
           'k--', label='O(h^2)')
plt.title('Central difference error vs h')
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('/Users/mera/Documents/tuition/CM/lab6/lab6_fig2.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# TASK 3: Numerical integration
# =====================================================================

def rectangle_left(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))

def rectangle_right(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))

def rectangle_mid(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5 * f(a) + np.sum(f(x[1:-1])) + 0.5 * f(b))

def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 1:
            result += 4 * f(x[i])
        else:
            result += 2 * f(x[i])
    return h / 3 * result

a, b = 1.0, 3.0
exact = 3.0 * np.log(3.0) - 2.0  # integral of ln(x) from 1 to 3
n_values = [2, 4, 8, 16, 32, 64]

print("\n--- TASK 3: Numerical integration ---")
print(f"Integral of ln(x) from 1 to 3 = {exact:.8f}")
print(f"{'n':<6} {'Left rect.':<12} {'Right rect.':<12} {'Mid rect.':<12} {'Trapezoid':<12} {'Simpson':<12}")
print("-" * 75)

for n in n_values:
    il = rectangle_left(f, a, b, n)
    ir = rectangle_right(f, a, b, n)
    im = rectangle_mid(f, a, b, n)
    it = trapezoidal(f, a, b, n)
    isim = simpson(f, a, b, n)
    print(f"{n:<6} {il:<12.8f} {ir:<12.8f} {im:<12.8f} {it:<12.8f} {isim:<12.8f}")

# =====================================================================
# TASK 4: Error comparison
# =====================================================================

print("\n--- TASK 4: Integration errors ---")
print(f"{'n':<6} {'Left err':<15} {'Right err':<15} {'Mid err':<15} {'Trap err':<15} {'Simpson err':<15}")
print("-" * 85)

errors_left = []
errors_right = []
errors_mid = []
errors_trap = []
errors_simp = []
for n in n_values:
    il = rectangle_left(f, a, b, n)
    ir = rectangle_right(f, a, b, n)
    im = rectangle_mid(f, a, b, n)
    it = trapezoidal(f, a, b, n)
    isim = simpson(f, a, b, n)
    el = abs(il - exact)
    er = abs(ir - exact)
    em = abs(im - exact)
    et = abs(it - exact)
    es = abs(isim - exact)
    errors_left.append(el)
    errors_right.append(er)
    errors_mid.append(em)
    errors_trap.append(et)
    errors_simp.append(es)
    print(f"{n:<6} {el:<15.2e} {er:<15.2e} {em:<15.2e} {et:<15.2e} {es:<15.2e}")

# Наблюдаемый порядок сходимости p = log2(err(n) / err(2n))
print("\nObserved convergence order p = log2(err_n / err_2n):")
print(f"{'n->2n':<8} {'Left':<8} {'Right':<8} {'Mid':<8} {'Trap':<8} {'Simpson':<8}")
for i in range(len(n_values) - 1):
    p = [np.log2(e[i] / e[i + 1]) for e in
         (errors_left, errors_right, errors_mid, errors_trap, errors_simp)]
    print(f"{n_values[i]}->{n_values[i+1]:<4} " + " ".join(f"{v:<8.2f}" for v in p))

plt.figure(figsize=(8, 5))
plt.loglog(n_values, errors_left, 'o-', label='Left rectangles')
plt.loglog(n_values, errors_right, 's-', label='Right rectangles')
plt.loglog(n_values, errors_mid, 'd-', label='Mid rectangles')
plt.loglog(n_values, errors_trap, '^-', label='Trapezoidal')
plt.loglog(n_values, errors_simp, '*-', label='Simpson')
plt.title('Comparison of integration errors')
plt.xlabel('Number of intervals n')
plt.ylabel('Absolute error')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('/Users/mera/Documents/tuition/CM/lab6/lab6_fig3.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ALL TASKS COMPLETED")
print("=" * 60)
