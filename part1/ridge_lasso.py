import numpy as np

def ridge_fit(X, y, alpha=1.0):
    """
    Cài đặt thuật toán Ridge Regression thuần bằng numpy
    Công thức: beta_hat = (X^T * X + alpha * I)^(-1) * X^T * y
    """
    n_features = X.shape[1]
    
    # Ma trận đơn vị (Identity Matrix)
    I = np.eye(n_features)
    # Không phạt hệ số chặn (intercept) nằm ở cột đầu tiên
    I[0, 0] = 0 
    
    X_T = X.T
    # Tính toán beta_hat theo công thức Ridge
    beta_hat = np.linalg.inv(X_T.dot(X) + alpha * I).dot(X_T).dot(y)
    
    return beta_hat

if __name__ == "__main__":
    print("Đã cài đặt thành công thuật toán Ridge Regression bằng công thức đại số tuyến tính.")