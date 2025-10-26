import numpy as np
import matplotlib.pyplot as plt

# ========================
# DATA
# ========================
I = np.array([0.0480, 0.0918, 0.1375, 0.19, 0.23, 0.28, 0.33, 0.37, 0.42, 0.46])
V = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sigma_x = np.array([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
sigma_y = np.full_like(sigma_x, 1e-12)  # σy extremely small to keep York exact

# ========================
# YORK ALGORITHM
# ========================
def york_regression(x, y, x_err, y_err):
    w_x = 1.0 / (x_err * x_err)
    w_y = 1.0 / (y_err * y_err)
    alpha = np.sqrt(w_x * w_y)

    b = np.sum(w_x * x * y) / np.sum(w_x * x * x)

    for _ in range(100):
        w = (w_x * w_y) / (b * b * w_y + w_x - 2.0 * b * alpha)
        x_bar = np.sum(w * x) / np.sum(w)
        y_bar = np.sum(w * y) / np.sum(w)
        U = x - x_bar
        Vv = y - y_bar
        beta = w * (U / w_y + b * Vv / w_x - (b * U + Vv) * alpha / (w_x * w_y))
        b = np.sum(w * beta * Vv) / np.sum(w * beta * U)

    a = y_bar - b * x_bar
    sb = np.sqrt(1.0 / np.sum(w * U * U))
    sa = np.sqrt((1.0 / np.sum(w)) + x_bar * x_bar * sb * sb)
    return b, a, sb, sa

# Run York
m, b, m_err, b_err = york_regression(I, V, sigma_x, sigma_y)

# Worst-case slopes
m_min = m - m_err
m_max = m + m_err

# ========================
# PRINT RESULTS
# ========================
print("\n--- YORK REGRESSION RESULTS ---")
print(f"Slope (m) = {m:.6f} ± {m_err:.6f}")
print(f"Intercept (b) = {b:.6f} ± {b_err:.6f}")
print(f"Worst-case slopes: m_min = {m_min:.6f}, m_max = {m_max:.6f}")

# ========================
# PLOT
# ========================
I_line = np.linspace(min(I), max(I), 100)
V_fit = m * I_line + b
V_min = m_min * I_line + b
V_max = m_max * I_line + b

plt.figure()
plt.errorbar(I, V, xerr=sigma_x, fmt='o', label="data")
plt.plot(I_line, V_fit, 'r', label="York fit")
plt.fill_between(I_line, V_min, V_max, color='gray', alpha=0.3, label="slope band")
plt.xlabel("Current I (A)")
plt.ylabel("Voltage V (V)")
plt.legend()
plt.grid(True)
plt.show()