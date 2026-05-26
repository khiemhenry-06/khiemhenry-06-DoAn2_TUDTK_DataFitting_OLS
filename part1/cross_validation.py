import numpy as np

def kfold_cv_split(X, y, k=5, random_seed=None):
    """
    Chia dữ liệu thành k fold để Cross Validation.
    Cài đặt thủ công bằng numpy, không dùng sklearn.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Chia đều các phần tử vào k fold
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((train_indices, test_indices))
        current = stop
        
    return folds

# Test nhanh hàm chạy ổn định
if __name__ == "__main__":
    X_test = np.arange(10).reshape(5, 2)
    y_test = np.arange(5)
    folds = kfold_cv_split(X_test, y_test, k=2, random_seed=42)
    print("Đã cài đặt thành công thuật toán K-Fold CV thủ công.")