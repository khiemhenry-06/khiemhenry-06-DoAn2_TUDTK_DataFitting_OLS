import numpy as np

def kfold_cv_split(X, y, k=5, random_seed=None):
    """
    Chia dữ liệu thành k fold để Cross Validation bằng numpy.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
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

if __name__ == "__main__":
    # Minh hoạ K-Fold với dữ liệu giả lập
    X_sim = np.random.randn(15, 2)
    y_sim = np.random.randn(15)
    
    print("--- Demo chia dữ liệu K-Fold (K=3) ---")
    folds = kfold_cv_split(X_sim, y_sim, k=3, random_seed=42)
    
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}:")
        print(f"  Train indices ({len(train_idx)}): {train_idx}")
        print(f"  Test indices ({len(test_idx)}): {test_idx}\n")
