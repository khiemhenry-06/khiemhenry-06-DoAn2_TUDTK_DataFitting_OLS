"""OLS implementation from scratch.

Provides: ols_fit, hat_matrix, rss, tss, r2_score, vif

Accepts numpy arrays or pandas DataFrame/Series.

Module này hiện thực hóa OLS hoàn toàn bằng NumPy (không dùng sklearn/statsmodels).
Bao gồm các hàm để tính ma trận Hat, RSS/TSS/R^2, ước lượng phương sai phần dư,
kiểm định t/p-value cho từng hệ số, và VIF để kiểm tra đa cộng tuyến.
Tất cả hàm chấp nhận đầu vào là numpy array hoặc pandas DataFrame/Series.
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

    Trả về ma trận 'hat' H sao cho y_hat = H y.
    Nếu bạn đã truyền vào ma trận thiết kế có cột intercept thì giữ nguyên.
    Kết quả là ma trận kích thước n x n, đối xứng và có tính chất idempotent (H^2 = H).
    """
    X = _ensure_numpy(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    XtX = X.T @ X
    inv = np.linalg.pinv(XtX)
    H = X @ inv @ X.T
    return H


def check_idempotent(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Kiểm tra tính idempotent của ma trận H.

    Trả về True nếu H @ H xấp xỉ bằng H trong sai số `tol`.
    """
    H = _ensure_numpy(H)
    return np.allclose(H @ H, H, atol=tol)


def rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Residual Sum of Squares (RSS).

    Tổng bình phương các phần dư (y - y_hat). Dùng để đánh giá mức sai số của mô hình trên dữ liệu quan sát.
    """
    y = _ensure_numpy(y).reshape(-1)
    y_hat = _ensure_numpy(y_hat).reshape(-1)
    return float(np.sum((y - y_hat) ** 2))


def tss(y: np.ndarray) -> float:
    """Total Sum of Squares (TSS).

    Tổng bình phương sai khác của y so với trung bình của y.
    Dùng làm chuẩn để so sánh với RSS khi tính R^2.
    """
    y = _ensure_numpy(y).reshape(-1)
    return float(np.sum((y - np.mean(y)) ** 2))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Phần trăm phương sai của y được mô hình giải thích (giá trị trong [0,1]).
    """
    return 1.0 - rss(y, y_hat) / tss(y)


def model_metrics(y, y_hat, p: int) -> Dict[str, float]:
    """Tính các chỉ số đánh giá mô hình hồi quy.

    Tham số:
    - `y`: vector quan sát thực tế
    - `y_hat`: vector dự đoán từ mô hình
    - `p`: số tham số trong mô hình (bao gồm intercept nếu có)

    Trả về dictionary gồm: RSS, TSS, R2, R2_adj, F_stat, p_value_F, n, p.
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
    """Ước lượng OLS bằng phương trình bình phương nhỏ nhất.
    Tham số:
    - `X`, `y`: dữ liệu (X có thể là numpy array hoặc pandas DataFrame)
    - `add_intercept`: nếu True, hàm sẽ thêm cột 1 làm hệ số chặn

    Hàm trả về một dict chứa các kết quả chính như `beta`, `y_hat`, `residuals`,
    các chỉ số RSS/TSS/R2, ước lượng phương sai phần dư `sigma2`, ma trận hiệp
    phương sai của `beta` (`cov`), sai số chuẩn (`se`), cùng thống kê t và p-value.
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
    """Tính các giá trị suy diễn cho hệ số hồi quy.

    Tham số:
    - `X`: ma trận thiết kế (có thể đã có cột intercept hoặc chưa)
    - `y`: vector quan sát (dùng để xác định kích thước mẫu/df)
    - `beta_hat`: vector hệ số ước lượng
    - `sigma2`: ước lượng phương sai phần dư

    Trả về `pandas.DataFrame` gồm các cột: `beta`, `se`, `t`, `p`, `ci_lower`, `ci_upper`.
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
    """Tính hệ số VIF (Variance Inflation Factor) cho từng biến giải thích.

    Với mỗi biến X_j (ngoại trừ intercept), hồi quy X_j trên các
    biến còn lại để lấy R_j^2 rồi tính VIF_j = 1 / (1 - R_j^2). Nếu `add_intercept`
    là True thì hàm sẽ bỏ qua cột chặn (gán NaN cho vị trí đó).
    Trả về mảng VIF và từ điển lưu R^2 tương ứng.
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
