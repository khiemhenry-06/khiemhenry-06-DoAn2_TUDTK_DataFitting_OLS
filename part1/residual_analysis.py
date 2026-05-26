import os

# Đỗ Trung Kiên
"""
=======================================================================
  (a) Trình bày lý thuyết từng biểu đồ
  (b) Cài đặt hàm residual_plots từ đầu bằng NumPy/Matplotlib
  (c) Minh họa bằng dữ liệu giả lập (synthetic data)
  (d) Kiểm chứng bằng statsmodels

4 biểu đồ:
  1. Residuals vs Fitted  → kiểm tra tính tuyến tính
  2. Normal Q-Q Plot      → kiểm tra phân phối chuẩn
  3. Scale-Location       → kiểm tra phương sai đồng đều
  4. Cook's Distance      → phát hiện điểm ảnh hưởng lớn
=======================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ── Kiểm chứng bằng statsmodels (chỉ để so sánh, không dùng để cài đặt)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# PHẦN 1 — LÝ THUYẾT
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  LÝ THUYẾT: PHÂN TÍCH PHẦN DƯ (RESIDUAL ANALYSIS)")
print("=" * 65)
print("""
Sau khi fit mô hình OLS: y = Xβ + ε, ta có:
  ŷ = X·β̂           (fitted values)
  ê = y - ŷ          (residuals — phần dư)

Phân tích phần dư kiểm tra 4 giả định Gauss-Markov:

  GM3. Ngoại sinh:       E[ε|X] = 0
       → Kiểm tra: Residuals vs Fitted (biểu đồ 1)

  GM5. Phân phối chuẩn: ε ~ N(0, σ²I)
       → Kiểm tra: Q-Q Plot (biểu đồ 2)

  GM4. Đồng phương sai:  Var(ε|X) = σ²I
       → Kiểm tra: Scale-Location (biểu đồ 3)

  Điểm ảnh hưởng lớn (Influential points):
       → Kiểm tra: Cook's Distance (biểu đồ 4)

  Phần dư chuẩn hóa:
       eᵢ_std = eᵢ / (σ̂ · √(1 - hᵢᵢ))
  với hᵢᵢ là phần tử thứ i trên đường chéo của Hat Matrix H = X(XᵀX)⁻¹Xᵀ

  Cook's Distance:
       Dᵢ = (eᵢ_std)² · hᵢᵢ / (p · (1 - hᵢᵢ))
  Dᵢ > 4/n → điểm có ảnh hưởng lớn đến ước lượng β̂
""")


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 2 — TẠO DỮ LIỆU GIẢ LẬP (Synthetic Data)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  TẠO DỮ LIỆU GIẢ LẬP")
print("=" * 65)

n, p = 200, 3
X_raw = np.random.randn(n, p)
X     = np.column_stack([np.ones(n), X_raw])  # thêm cột 1 (intercept)
beta_true = np.array([2.0, 1.5, -0.8, 0.5])
epsilon   = np.random.normal(0, 1.2, n)
y         = X @ beta_true + epsilon

# Thêm 5 điểm outlier để biểu đồ minh họa rõ hơn
outlier_idx = np.random.choice(n, size=5, replace=False)
y[outlier_idx] += np.random.choice([-8, 8], size=5)

print(f"  n = {n} quan sát  |  p = {p} biến (+ intercept)")
print(f"  β thực = {beta_true}")
print(f"  Thêm {len(outlier_idx)} điểm outlier tại index: {sorted(outlier_idx)}")


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 3 — CÀI ĐẶT HÀM TỪ ĐẦU
# ═══════════════════════════════════════════════════════════════════════

def ols_fit(X, y):
    """
    Ước lượng OLS: β̂ = (XᵀX)⁻¹ Xᵀy
    Trả về: beta_hat, y_hat, residuals, sigma2_hat
    """
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat    = X @ beta_hat
    residuals = y - y_hat
    n, p_plus1 = X.shape
    sigma2_hat = (residuals @ residuals) / (n - p_plus1)
    return beta_hat, y_hat, residuals, sigma2_hat


def hat_matrix_diag(X):
    """
    Tính đường chéo của Hat Matrix H = X(XᵀX)⁻¹Xᵀ
    Chỉ cần đường chéo hᵢᵢ, không cần tính toàn bộ ma trận (tránh O(n²) bộ nhớ)
    hᵢᵢ = xᵢᵀ (XᵀX)⁻¹ xᵢ
    """
    XtX_inv = np.linalg.inv(X.T @ X)
    # hᵢᵢ = diag(X (XᵀX)⁻¹ Xᵀ)
    H_diag = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    return H_diag


def standardized_residuals(residuals, sigma2_hat, h_diag):
    """
    Phần dư chuẩn hóa (Internally Studentized Residuals):
    eᵢ_std = eᵢ / (σ̂ · √(1 - hᵢᵢ))
    """
    sigma_hat = np.sqrt(sigma2_hat)
    denom = sigma_hat * np.sqrt(np.maximum(1 - h_diag, 1e-10))
    return residuals / denom


def cooks_distance(std_resid, h_diag, p):
    """
    Cook's Distance:
    Dᵢ = (eᵢ_std)² · hᵢᵢ / (p · (1 - hᵢᵢ))
    """
    denom = p * np.maximum(1 - h_diag, 1e-10)
    return (std_resid ** 2) * h_diag / denom


def residual_plots(X, y, beta_hat, title_suffix=""):
    """
    Vẽ 4 biểu đồ phân tích phần dư chuẩn:
      1. Residuals vs Fitted
      2. Normal Q-Q Plot
      3. Scale-Location
      4. Cook's Distance

    Tham số:
      X        : ma trận đặc trưng (đã có cột intercept)
      y        : vector mục tiêu
      beta_hat : hệ số OLS đã ước lượng
      title_suffix : chuỗi thêm vào tiêu đề (tuỳ chọn)
    """
    n, p_plus1 = X.shape
    p          = p_plus1 - 1   # số biến (trừ intercept)

    # ── Tính các đại lượng cần thiết ────────────────────────────────
    y_hat     = X @ beta_hat
    residuals = y - y_hat
    sigma2    = (residuals @ residuals) / (n - p_plus1)
    h_diag    = hat_matrix_diag(X)
    std_resid = standardized_residuals(residuals, sigma2, h_diag)
    cooks_d   = cooks_distance(std_resid, h_diag, p_plus1)
    sqrt_abs  = np.sqrt(np.abs(std_resid))

    # ── Vẽ ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    suf = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Phân tích Phần dư (Residual Analysis){suf}",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── 1. Residuals vs Fitted ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_hat, residuals, alpha=0.5, s=20,
                color='steelblue', edgecolors='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)

    # Làm trơn LOWESS thủ công bằng moving average theo bins
    sort_idx = np.argsort(y_hat)
    y_s, r_s = y_hat[sort_idx], residuals[sort_idx]
    w = max(n // 8, 5)
    smooth_x = [y_s[i:i+w].mean() for i in range(0, n - w, w // 2)]
    smooth_y = [r_s[i:i+w].mean() for i in range(0, n - w, w // 2)]
    ax1.plot(smooth_x, smooth_y, color='orange', linewidth=2,
             label='Moving avg (làm trơn)')

    ax1.set_xlabel("Fitted values (ŷ)", fontsize=10)
    ax1.set_ylabel("Residuals (ê = y - ŷ)", fontsize=10)
    ax1.set_title("1. Residuals vs Fitted\n→ Kiểm tra E[ε|X] = 0 (GM3)",
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.text(0.97, 0.04, "Tốt: điểm phân tán đều quanh đường 0",
             transform=ax1.transAxes, fontsize=8, ha='right',
             color='gray', style='italic')

    # ── 2. Normal Q-Q Plot ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r_val) = stats.probplot(std_resid, dist="norm")
    ax2.plot(osm, osr, 'o', alpha=0.5, markersize=5, color='steelblue',
             label='Quantiles thực tế')
    ax2.plot(osm, slope * np.array(osm) + intercept,
             'r--', linewidth=1.5, label='Đường lý thuyết chuẩn')
    ax2.set_xlabel("Theoretical Quantiles (phân vị lý thuyết)", fontsize=10)
    ax2.set_ylabel("Sample Quantiles (phân vị thực tế)", fontsize=10)
    ax2.set_title("2. Normal Q-Q Plot\n→ Kiểm tra ε ~ N(0, σ²) (GM5)",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.text(0.97, 0.04, f"R² QQ = {r_val**2:.4f} (gần 1 = chuẩn)",
             transform=ax2.transAxes, fontsize=8, ha='right',
             color='gray', style='italic')

    # ── 3. Scale-Location ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_hat, sqrt_abs, alpha=0.5, s=20,
                color='steelblue', edgecolors='none')

    smooth_y3 = [sqrt_abs[sort_idx][i:i+w].mean() for i in range(0, n - w, w // 2)]
    ax3.plot(smooth_x, smooth_y3, color='orange', linewidth=2)

    ax3.set_xlabel("Fitted values (ŷ)", fontsize=10)
    ax3.set_ylabel("√|Std Residuals|", fontsize=10)
    ax3.set_title("3. Scale-Location\n→ Kiểm tra Var(ε|X) = σ²I (GM4)",
                  fontsize=10, fontweight='bold')
    ax3.text(0.97, 0.04, "Tốt: đường cam nằm ngang (phương sai đều)",
             transform=ax3.transAxes, fontsize=8, ha='right',
             color='gray', style='italic')

    # ── 4. Cook's Distance ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    threshold = 4 / n
    colors_bar = ['red' if d > threshold else 'steelblue' for d in cooks_d]
    ax4.bar(range(n), cooks_d, color=colors_bar, width=1.0, alpha=0.7)
    ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
                label=f"Ngưỡng 4/n = {threshold:.4f}")

    n_influential = (cooks_d > threshold).sum()
    for i in np.where(cooks_d > threshold)[0][:5]:  # label tối đa 5 điểm
        ax4.annotate(f"[{i}]", (i, cooks_d[i]),
                     textcoords="offset points", xytext=(0, 4),
                     fontsize=7, color='red', ha='center')

    ax4.set_xlabel("Chỉ số quan sát (index)", fontsize=10)
    ax4.set_ylabel("Cook's Distance (Dᵢ)", fontsize=10)
    ax4.set_title("4. Cook's Distance\n→ Phát hiện điểm ảnh hưởng lớn",
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.text(0.97, 0.04, f"Phát hiện {n_influential} điểm ảnh hưởng lớn (màu đỏ)",
             transform=ax4.transAxes, fontsize=8, ha='right',
             color='red' if n_influential > 0 else 'gray', style='italic')

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'report', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'residual_analysis_part1.png')
    plt.savefig(output_path, dpi=150,
          bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n  ✓ Đã lưu: {output_path}")

    return {
        'y_hat': y_hat, 'residuals': residuals,
        'std_residuals': std_resid, 'h_diag': h_diag,
        'cooks_d': cooks_d, 'n_influential': n_influential,
        'sigma2': sigma2
    }


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 4 — CHẠY VÀ IN KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  CHẠY HÀM residual_plots (tự cài đặt)")
print("=" * 65)

beta_hat, y_hat, residuals, sigma2_hat = ols_fit(X, y)
print(f"  β̂  = {np.round(beta_hat, 4)}")
print(f"  β  = {beta_true}  (giá trị thực)")
print(f"  σ̂² = {sigma2_hat:.4f}  (σ² thực = {1.2**2:.4f})")

diag_results = residual_plots(X, y, beta_hat,
                              title_suffix="Dữ liệu giả lập (n=200, p=3)")

print(f"\n  Số điểm ảnh hưởng lớn (Cook's D > 4/n): {diag_results['n_influential']}")

# Kiểm định Shapiro-Wilk cho phần dư
sw_stat, sw_p = stats.shapiro(diag_results['std_residuals'])
print(f"  Shapiro-Wilk test: W={sw_stat:.4f}, p={sw_p:.4f}")
print(f"  → {'Không bác bỏ H₀ (chuẩn) ✓' if sw_p > 0.05 else 'Bác bỏ H₀ — phần dư không chuẩn (do outlier) ✗'}")


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 5 — KIỂM CHỨNG BẰNG STATSMODELS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  KIỂM CHỨNG BẰNG STATSMODELS")
print("=" * 65)

sm_model    = sm.OLS(y, X).fit()
sm_influence = OLSInfluence(sm_model)
sm_std_resid = sm_influence.resid_studentized_internal
sm_h_diag    = sm_influence.hat_matrix_diag
sm_cooks_d   = sm_influence.cooks_distance[0]

# So sánh
h_diff    = np.max(np.abs(diag_results['h_diag'] - sm_h_diag))
resid_diff = np.max(np.abs(diag_results['std_residuals'] - sm_std_resid))
cooks_diff = np.max(np.abs(diag_results['cooks_d'] - sm_cooks_d))

print(f"  Max sai lệch leverage (hᵢᵢ):        {h_diff:.2e}")
print(f"  Max sai lệch std residuals:          {resid_diff:.2e}")
print(f"  Max sai lệch Cook's Distance:        {cooks_diff:.2e}")
print(f"  → {'Kết quả khớp hoàn toàn ✓' if max(h_diff, resid_diff, cooks_diff) < 1e-6 else 'Có sai lệch nhỏ (chấp nhận được)'}")


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 6 — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  UNIT TESTS")
print("=" * 65)

def test_hat_matrix_properties():
    """
    Kiểm tra tính chất của Hat Matrix H:
    - H² = H (idempotent): hᵢᵢ = Σⱼ H[i,j]² = hᵢᵢ²  (không trực tiếp nhưng dùng trace)
    - 0 ≤ hᵢᵢ ≤ 1
    - Σᵢ hᵢᵢ = rank(X) = p+1
    """
    X_test = np.column_stack([np.ones(50), np.random.randn(50, 2)])
    h      = hat_matrix_diag(X_test)
    n_t, p1 = X_test.shape

    assert np.all(h >= -1e-10), "hᵢᵢ phải >= 0"
    assert np.all(h <= 1 + 1e-10), "hᵢᵢ phải <= 1"
    assert abs(h.sum() - p1) < 1e-6, f"Σhᵢᵢ = {h.sum():.6f} ≠ {p1}"
    print("  ✓ test_hat_matrix_properties: 0≤hᵢᵢ≤1, Σhᵢᵢ=p+1")

def test_residuals_orthogonal():
    """
    OLS residuals phải trực giao với cột của X: Xᵀê = 0
    """
    X_test  = np.column_stack([np.ones(50), np.random.randn(50, 2)])
    y_test  = np.random.randn(50)
    beta, y_hat, resid, _ = ols_fit(X_test, y_test)
    xt_resid = X_test.T @ resid
    assert np.max(np.abs(xt_resid)) < 1e-8, f"Xᵀê ≠ 0: max={np.max(np.abs(xt_resid))}"
    print("  ✓ test_residuals_orthogonal: Xᵀê = 0")

def test_cooks_distance_zero_outlier():
    """
    Với dữ liệu hoàn hảo (không nhiễu), Cook's Distance phải rất nhỏ.
    """
    X_test  = np.column_stack([np.ones(30), np.linspace(0, 1, 30)])
    beta    = np.array([1.0, 2.0])
    y_test  = X_test @ beta  # không nhiễu
    _, _, resid, sigma2 = ols_fit(X_test, y_test)
    h       = hat_matrix_diag(X_test)
    s_resid = standardized_residuals(resid, sigma2 + 1e-10, h)
    cd      = cooks_distance(s_resid, h, 2)
    assert cd.max() < 1e-4, f"Cook's D phải gần 0 khi không có nhiễu, nhưng max={cd.max()}"
    print("  ✓ test_cooks_distance_zero_outlier: Dᵢ ≈ 0 khi không nhiễu")

def test_standardized_residuals_scale():
    """
    Phần dư chuẩn hóa phải có trung bình gần 0 và std gần 1
    (với n đủ lớn và không outlier).
    """
    n_t   = 500
    X_test = np.column_stack([np.ones(n_t), np.random.randn(n_t, 3)])
    y_test = X_test @ np.array([1, 0.5, -0.3, 0.8]) + np.random.randn(n_t)
    beta, _, resid, sigma2 = ols_fit(X_test, y_test)
    h      = hat_matrix_diag(X_test)
    s_resid = standardized_residuals(resid, sigma2, h)
    assert abs(s_resid.mean()) < 0.1, f"Mean std_resid = {s_resid.mean():.4f} ≠ ~0"
    assert abs(s_resid.std() - 1) < 0.2, f"Std std_resid = {s_resid.std():.4f} ≠ ~1"
    print("  ✓ test_standardized_residuals_scale: mean≈0, std≈1")

test_hat_matrix_properties()
test_residuals_orthogonal()
test_cooks_distance_zero_outlier()
test_standardized_residuals_scale()
print("\n  Tất cả Unit Tests đã PASS ✓")
print("\nXONG! ✓")