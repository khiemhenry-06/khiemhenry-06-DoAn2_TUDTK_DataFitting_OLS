import numpy as np
import matplotlib.pyplot as plt

# 1. Khởi tạo tham số thật
np.random.seed(42)
N, p = 100, 2 # 100 mẫu, 2 biến
X = np.random.randn(N, p)
X = np.c_[np.ones(N), X] # Thêm cột intercept
beta_true = np.array([2.5, 1.5, -3.0])

# 2. Cấu hình mô phỏng
M = 1000
beta_estimates = []

# 3. Lạy lặp Monte Carlo
for _ in range(M):
    epsilon = np.random.normal(0, 1, N)
    y = X.dot(beta_true) + epsilon
    
    # Tính OLS: (X^T X)^(-1) X^T y
    X_T = X.T
    beta_hat = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
    beta_estimates.append(beta_hat)

beta_estimates = np.array(beta_estimates)

# 4. Trực quan hóa phân phối của hệ số beta_1
plt.hist(beta_estimates[:, 1], bins=30, edgecolor='k', alpha=0.7)
plt.axvline(beta_true[1], color='red', linestyle='dashed', linewidth=2, label=f'True $\\beta_1$: {beta_true[1]}')
plt.title("Phân phối của ước lượng OLS qua mô phỏng Monte Carlo")
plt.legend()
plt.show()