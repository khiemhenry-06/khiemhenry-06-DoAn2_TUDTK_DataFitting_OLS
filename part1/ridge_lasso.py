import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def ridge_fit(X, y, alpha=1.0):
    """
    Cài đặt thuật toán Ridge Regression thuần bằng numpy
    Công thức: beta_hat = (X^T * X + alpha * I)^(-1) * X^T * y
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0 # Không phạt intercept
    
    X_T = X.T
    beta_hat = np.linalg.inv(X_T.dot(X) + alpha * I).dot(X_T).dot(y)
    return beta_hat

if __name__ == "__main__":
    # 1. Tạo dữ liệu giả lập (Synthetic data)
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X_sim = np.random.randn(n_samples, n_features)
    X_sim_with_intercept = np.c_[np.ones(n_samples), X_sim]
    true_coef = np.array([1.5, -2.0, 3.5, 0.0, 0.5])
    y_sim = X_sim.dot(true_coef) + np.random.randn(n_samples) * 0.5

    # 2. Kiểm chứng với thư viện sklearn
    alpha_test = 2.0
    beta_manual = ridge_fit(X_sim_with_intercept, y_sim, alpha=alpha_test)
    
    sk_ridge = Ridge(alpha=alpha_test, fit_intercept=True)
    sk_ridge.fit(X_sim, y_sim)
    beta_sklearn = np.insert(sk_ridge.coef_, 0, sk_ridge.intercept_)
    
    print("Hệ số Ridge (Cài đặt thủ công):", np.round(beta_manual, 4))
    print("Hệ số Ridge (Thư viện Sklearn):", np.round(beta_sklearn, 4))
    print("Khác biệt lớn nhất:", np.max(np.abs(beta_manual - beta_sklearn)))

    # 3. Vẽ biểu đồ Ridge Trace
    alphas = np.logspace(-2, 4, 100)
    coefs = []
    for a in alphas:
        # Lấy hệ số (bỏ intercept ở vị trí 0)
        coefs.append(ridge_fit(X_sim_with_intercept, y_sim, alpha=a)[1:])

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, coefs)
    plt.xscale('log')
    plt.xlabel('Siêu tham số Alpha (Log Scale)')
    plt.ylabel('Giá trị các hệ số (Coefficients)')
    plt.title('Ridge Trace: Sự co rút của các hệ số hồi quy theo Alpha')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()