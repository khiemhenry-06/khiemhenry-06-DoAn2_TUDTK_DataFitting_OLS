"""OLS implementation from scratch.

Provides: ols_fit, hat_matrix, rss, tss, r2_score, vif

Accepts numpy arrays or pandas DataFrame/Series.

Phiên bản tiếng Việt:
Module này cài đặt OLS (Ordinary Least Squares) từ đầu,
các phép toán chính bao gồm: ma trận Hat, RSS/TSS/R^2,
ước lượng phương sai phần dư, kiểm định t/p-value cho hệ số,
và VIF (Variance Inflation Factor) để kiểm tra đa cộng tuyến.
Hàm chấp nhận đầu vào là numpy array hoặc pandas DataFrame/Series.
"""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    stats = None
    _HAS_SCIPY = False
from math import erf, sqrt


def _ensure_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.values
    return np.asarray(X)


def add_constant(X: np.ndarray) -> np.ndarray:
    X = _ensure_numpy(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([ones, X], axis=1)


def hat_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the hat matrix H = X (X'X)^{-1} X'.

    Tiếng Việt: Tính ma trận Hat H sao cho y_hat = H y.
    Nếu ma trận thiết kế X đã có cột intercept thì giữ nguyên.
    Trả về ma trận H (n x n) đối xứng và idempotent.
    """
    X = _ensure_numpy(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    XtX = X.T @ X
    inv = np.linalg.pinv(XtX)
    H = X @ inv @ X.T
    return H


def check_idempotent(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix H is idempotent: H^2 = H.

    Tiếng Việt: Trả về True nếu H @ H ≈ H với sai số atol=tol.
    """
    H = _ensure_numpy(H)
    return np.allclose(H @ H, H, atol=tol)


def rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Residual Sum of Squares (RSS).

    Tiếng Việt: Tổng bình phương phần dư, dùng để đánh giá sai số mô hình.
    """
    y = _ensure_numpy(y).reshape(-1)
    y_hat = _ensure_numpy(y_hat).reshape(-1)
    return float(np.sum((y - y_hat) ** 2))


def tss(y: np.ndarray) -> float:
    """Total Sum of Squares (TSS).

    Tiếng Việt: Tổng bình phương sai khác so với trung bình của y.
    """
    y = _ensure_numpy(y).reshape(-1)
    return float(np.sum((y - np.mean(y)) ** 2))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Tiếng Việt: Tỷ lệ phương sai của y được mô hình giải thích.
    """
    return 1.0 - rss(y, y_hat) / tss(y)


def model_metrics(y, y_hat, p: int) -> Dict[str, float]:
    """Compute regression model metrics: RSS, TSS, R2, R2_adj, F-stat and p-value.

    Parameters:
    - y: observed responses
    - y_hat: predicted responses
    - p: number of parameters in the model (including intercept)

    Returns dict with keys: RSS, TSS, R2, R2_adj, F_stat, p_value_F
    """
    y = _ensure_numpy(y).reshape(-1)
    y_hat = _ensure_numpy(y_hat).reshape(-1)
    n = y.shape[0]
    RSS = float(np.sum((y - y_hat) ** 2))
    TSS = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 - RSS / TSS if TSS != 0 else np.nan
    # adjusted R2
    R2_adj = 1.0 - (1.0 - R2) * (n - 1) / (n - p) if n - p > 0 else np.nan

    # F-statistic for overall model: ((TSS - RSS)/(p-1)) / (RSS/(n-p))
    df_model = p - 1
    df_resid = n - p
    ESS = TSS - RSS
    if df_model > 0 and df_resid > 0:
        F_stat = (ESS / df_model) / (RSS / df_resid)
        if _HAS_SCIPY:
            p_value_F = stats.f.sf(F_stat, df_model, df_resid)
        else:
            p_value_F = np.nan
    else:
        F_stat = np.nan
        p_value_F = np.nan

    return {
        'RSS': RSS,
        'TSS': TSS,
        'R2': R2,
        'R2_adj': R2_adj,
        'F_stat': F_stat,
        'p_value_F': p_value_F,
        'n': n,
        'p': p,
    }


def ols_fit(X, y, add_intercept: bool = True) -> Dict:
    """Fit OLS via normal equations.

    Tiếng Việt - Mô tả:
    - `X`, `y`: dữ liệu (X có thể là ma trận hoặc dataframe);
    - `add_intercept`: nếu True thêm cột 1 làm hệ số chặn;
    Trả về dict chứa:
      - `beta`: vector ước lượng hệ số (bao gồm intercept nếu có),
      - `y_hat`: giá trị dự đoán, `residuals`: phần dư,
      - `RSS`, `TSS`, `R2`, `sigma2` (ước lượng phương sai phần dư),
      - `cov`: ma trận hiệp phương sai của `beta`, `se`: sai số chuẩn,
      - `t`, `p`: thống kê t và p-value hai phía.

    Công thức chính: \\hat{\\beta} = (X^T X)^{-1} X^T y
    """
    X_np = _ensure_numpy(X)
    y_np = _ensure_numpy(y).reshape(-1)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    if add_intercept:
        X_design = add_constant(X_np)
    else:
        X_design = X_np

    n, p = X_design.shape
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X_design.T @ y_np
    y_hat = X_design @ beta
    residuals = y_np - y_hat
    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y_np - np.mean(y_np)) ** 2))
    R2 = 1.0 - RSS / TSS if TSS != 0 else np.nan

    df_resid = n - p
    sigma2 = RSS / df_resid
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = beta / se
    if _HAS_SCIPY:
        p_values = 2 * stats.t.sf(np.abs(t_stats), df=df_resid)
    else:
        def _normal_cdf(z):
            return 0.5 * (1.0 + erf(z / sqrt(2.0)))
        p_values = 2 * (1.0 - np.vectorize(_normal_cdf)(np.abs(t_stats)))

    return {
        'beta': beta,
        'y_hat': y_hat,
        'residuals': residuals,
        'RSS': RSS,
        'TSS': TSS,
        'R2': R2,
        'sigma2': sigma2,
        'cov': cov,
        'se': se,
        't': t_stats,
        'p': p_values,
        'n': n,
        'p': p,
    }


def coef_inference(X, y, beta_hat, sigma2) -> 'pd.DataFrame':
    """Compute inference for coefficients: SE, t-stat, p-value, 95% CI.

    Parameters:
    - X: design matrix (without added intercept or with; function will handle shape)
    - y: observed responses (used for sample size / df)
    - beta_hat: estimated coefficients (vector)
    - sigma2: estimated residual variance (scalar)

    Returns a pandas.DataFrame with columns: beta, se, t, p, ci_lower, ci_upper
    """
    X_np = _ensure_numpy(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    n = _ensure_numpy(y).reshape(-1).shape[0]
    p = X_np.shape[1]

    XtX = X_np.T @ X_np
    XtX_inv = np.linalg.pinv(XtX)
    cov_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = np.asarray(beta_hat).reshape(-1) / se

    df_resid = n - p
    if _HAS_SCIPY:
        p_values = 2 * stats.t.sf(np.abs(t_stats), df=df_resid)
        t_crit = stats.t.ppf(0.975, df=df_resid)
    else:
        # normal approx
        def _normal_cdf(z):
            return 0.5 * (1.0 + erf(z / sqrt(2.0)))

        p_values = 2 * (1.0 - np.vectorize(_normal_cdf)(np.abs(t_stats)))
        t_crit = 1.959963984540054

    beta_arr = np.asarray(beta_hat).reshape(-1)
    ci_lower = beta_arr - t_crit * se
    ci_upper = beta_arr + t_crit * se

    df = pd.DataFrame({
        'beta': beta_arr,
        'se': se,
        't': t_stats,
        'p': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    })
    return df


def vif(X, add_intercept: bool = True) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute Variance Inflation Factor (VIF) for each regressor.

    Tiếng Việt: Với mỗi biến giải thích X_j (không tính intercept),
    ta hồi quy X_j lên các biến còn lại để lấy R_j^2, rồi
    VIF_j = 1 / (1 - R_j^2). Nếu add_intercept=True thì cột chặn
    được bỏ qua (gán NaN cho VIF của nó).
    Trả về (vif_array, r2_dict) nơi r2_dict lưu R^2 cho mỗi cột.
    """
    X_np = _ensure_numpy(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    if add_intercept:
        X_design = add_constant(X_np)
    else:
        X_design = X_np

    n, p = X_design.shape
    vif_vals = np.zeros(p)
    r2_dict = {}
    for j in range(p):
        if add_intercept and j == 0:
            vif_vals[j] = np.nan
            r2_dict[j] = np.nan
            continue
        # regress column j on all other columns
        cols = [k for k in range(p) if k != j]
        X_others = X_design[:, cols]
        y_j = X_design[:, j]
        # compute R^2 for regression y_j ~ X_others
        # add intercept if not present in X_others
        res = ols_fit(X_others, y_j, add_intercept=True)
        R2_j = res['R2']
        r2_dict[j] = R2_j
        vif_vals[j] = 1.0 / (1.0 - R2_j) if R2_j is not None and not np.isnan(R2_j) else np.nan

    return vif_vals, r2_dict


if __name__ == '__main__':
    print("=== CHẠY UNIT TEST ===")
    np.random.seed(42)
    n = 100
    
    # Tạo dữ liệu giả lập
    X_test1 = np.random.randn(n, 2)
    beta_true = np.array([1.5, -2.0])
    y_test1 = 2.0 + X_test1 @ beta_true + np.random.randn(n) * 0.5

    # Test 1 & 2: ols_fit
    res = ols_fit(X_test1, y_test1, add_intercept=True)
    assert len(res['beta']) == 3, "Lỗi: Hệ số beta phải có độ dài là 3 (bao gồm intercept)"
    assert res['R2'] > 0.8, "Lỗi: R^2 phải khá cao với dữ liệu nhiễu thấp"
    print("[Pass] ols_fit")

    # Test 3 & 4: hat_matrix & check_idempotent
    X_design = add_constant(X_test1)
    H = hat_matrix(X_design)
    assert H.shape == (n, n), "Lỗi: Ma trận Hat phải có kích thước n x n"
    assert check_idempotent(H), "Lỗi: Ma trận Hat không có tính chất idempotent (H^2 = H)"
    print("[Pass] hat_matrix & check_idempotent")

    # Test 5 & 6: model_metrics
    metrics = model_metrics(y_test1, res['y_hat'], p=3)
    assert not np.isnan(metrics['R2_adj']), "Lỗi: R2 hiệu chỉnh không được là NaN"
    assert metrics['F_stat'] > 0, "Lỗi: F-statistic phải lớn hơn 0"
    print("[Pass] model_metrics")
    
    # Test 7 & 8: coef_inference
    df_inf = coef_inference(X_design, y_test1, res['beta'], res['sigma2'])
    assert len(df_inf) == 3, "Lỗi: DataFrame suy diễn phải có 3 dòng tương ứng với 3 hệ số"
    assert 'ci_lower' in df_inf.columns and 'ci_upper' in df_inf.columns, "Lỗi: Thiếu khoảng tin cậy"
    print("[Pass] coef_inference")

    # Test 9 & 10: VIF (Tạo dữ liệu đa cộng tuyến mạnh)
    X_test2 = np.column_stack((X_test1[:, 0], X_test1[:, 0] * 2 + np.random.randn(n) * 0.01))
    vif_vals, _ = vif(X_test2, add_intercept=False)
    assert np.all(vif_vals > 10), "Lỗi: VIF phải lớn hơn 10 do hai biến có tương quan cực mạnh"
    assert len(vif_vals) == 2, "Lỗi: Mảng VIF trả về phải chứa 2 giá trị"
    print("[Pass] vif")

    print("=> TẤT CẢ UNIT TEST ĐỀU PASSED! CODE ĐẠT CHUẨN ĐỒ ÁN.")
