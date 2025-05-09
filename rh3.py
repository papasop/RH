import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi

# Step 1: Prime density π(x)/x
x_vals = np.linspace(100, 10000, 500)
logx_vals = np.log(x_vals)
pi_x = np.array([float(primepi(int(x))) for x in x_vals])
pi_norm = pi_x / x_vals

# Step 2: First 15 nontrivial zeta zeros as modal frequencies
t_n = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.4462477, 59.347044, 60.83177852, 65.11254405
])
N = len(t_n)
T = 100

# Step 3: Initialize modal amplitudes and phases
A_t = np.zeros((T, N))
theta_t = np.zeros((T, N))
H_t = np.zeros(T)
delta_L2 = np.zeros(T)
delta_xt = np.zeros((T, len(x_vals)))

np.random.seed(0)
A_t[0] = np.random.uniform(0.01, 0.1, N)
theta_t[0] = np.random.uniform(0, 2 * np.pi, N)

# Step 4: Modal field construction
def compute_rho(xlog, A, theta):
    rho = 1 / xlog
    for i in range(N):
        rho += A[i] * np.cos(t_n[i] * xlog + theta[i])
    return rho

def entropy(A):
    p = A / (np.sum(A) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

# Step 5: Evolve over time via entropy + residual feedback
lr = 0.01
for t in range(1, T):
    A_prev = A_t[t - 1]
    theta_prev = theta_t[t - 1]
    rho = compute_rho(logx_vals, A_prev, theta_prev)
    delta = pi_norm - rho

    A_new = np.zeros_like(A_prev)
    theta_new = np.zeros_like(theta_prev)

    for i in range(N):
        basis = np.cos(t_n[i] * logx_vals + theta_prev[i])
        grad_A = -2 * np.mean(delta * basis)
        A_new[i] = np.clip(A_prev[i] - lr * grad_A, 0, 0.2)

        grad_theta = 2 * A_prev[i] * np.mean(delta * np.sin(t_n[i] * logx_vals + theta_prev[i]))
        theta_new[i] = (theta_prev[i] - lr * grad_theta) % (2 * np.pi)

    A_t[t] = A_new
    theta_t[t] = theta_new
    H_t[t] = entropy(A_new)
    delta_xt[t] = delta
    delta_L2[t] = np.linalg.norm(delta)

# Step 6: Final modal snapshot and reconstruction
final_A = A_t[-1]
final_rho = compute_rho(logx_vals, final_A, theta_t[-1])
final_active_modes = np.where(final_A > 0.01)[0]
mse = np.mean((pi_norm - final_rho)**2)

# Step 7: Output results
print("Entropy collapse ΔH:", round(H_t[0] - H_t[-1], 4))
print("Residual norm collapse Δ||δ(x)||₂:", round(delta_L2[0] - delta_L2[-1], 4))
print("Final MSE:", round(mse, 6))
print("Activated ψ-path indices:", final_active_modes.tolist())

# Step 8: Optional plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(T), H_t, lw=2, label='Entropy H(t)')
plt.title('Entropy Collapse'); plt.xlabel('t'); plt.grid(True); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(T), delta_L2, lw=2, label='Residual Norm ||δ(x)||₂', color='orange')
plt.title('Residual Compression'); plt.xlabel('t'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
