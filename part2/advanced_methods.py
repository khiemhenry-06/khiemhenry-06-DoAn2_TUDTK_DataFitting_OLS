# Đỗ Trung Kiên

"""
=======================================================================
Đánh giá Mô hình & Kỹ thuật Nâng cao
=======================================================================
Nhiệm vụ:
  1. Tính MAE, RMSE, R² so sánh 3 mô hình
  2. Vẽ 4 biểu đồ phân tích phần dư (mô hình tốt nhất)
  3. Vẽ Feature Importance (Ridge coef)
  4. Kernel Ridge Regression — đúng công thức đề (+0.5 điểm)

Cách dùng:
  - Đặt file video_games_sales.csv và data_pipeline.py cùng thư mục
  - Chạy: python advanced_methods.py
=======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import warnings
import sys
import os
warnings.filterwarnings('ignore')

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 0 — ĐỌC VÀ XỬ LÝ DATA THẬT
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV     = os.path.join(BASE_DIR, "data", "video_games_sales.csv")
PIPELINE_DIR = BASE_DIR
IMAGE_DIR    = os.path.join(os.path.dirname(BASE_DIR), "report", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_report_figure(filename, **kwargs):
    output_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(output_path):
        os.remove(output_path)
    plt.savefig(output_path, **kwargs)

sys.path.insert(0, PIPELINE_DIR)
from data_pipeline import DataPipeline

print("=" * 60)
print("  BƯỚC 0: ĐỌC VÀ XỬ LÝ DATA")
print("=" * 60)

df_raw = pd.read_csv(DATA_CSV)
print(f"  Đọc xong: {df_raw.shape[0]:,} dòng x {df_raw.shape[1]} cột")

DROP_COLS = ['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
             'Developer', 'Rating', 'Year_of_Release']
df_raw = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
df_raw = df_raw.dropna(subset=['Global_Sales'])
print(f"  Sau khi lọc: {df_raw.shape[0]:,} dòng")

FEATURE_COLS = ['Platform', 'Genre', 'Publisher',
                'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
X_raw = df_raw[FEATURE_COLS].copy()
y     = df_raw['Global_Sales'].values

X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(y_train):,} mẫu  |  Test: {len(y_test):,} mẫu")

pipeline = DataPipeline(top_n_publishers=30)
X_train = pipeline.fit_transform(X_raw_train).values.astype(float)
X_test  = pipeline.transform(X_raw_test).values.astype(float)
feature_names_encoded = pipeline.feature_columns_
print(f"  Số đặc trưng sau encoding: {X_train.shape[1]}")
print(f"  Còn missing value? Train={np.isnan(X_train).any()} | Test={np.isnan(X_test).any()}")


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 0b — TRAIN 3 MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BƯỚC 0b: HUẤN LUYỆN 3 MÔ HÌNH")
print("=" * 60)

X_train_sm = sm.add_constant(X_train)
X_test_sm  = sm.add_constant(X_test)

# Model 1: OLS Full
ols_full = sm.OLS(y_train, X_train_sm).fit()
print(f"  OLS Full    — R² train: {ols_full.rsquared:.4f}")

# Model 2: OLS Stepwise Backward (loại biến p-value > 0.05)
def backward_stepwise_ols(X, y, threshold=0.05):
    cols = list(range(X.shape[1]))
    while True:
        model = sm.OLS(y, sm.add_constant(X[:, cols])).fit()
        pvals = model.pvalues[1:]
        max_p = pvals.max()
        if max_p > threshold:
            cols.pop(pvals.argmax())
        else:
            return model, cols

print("  Đang chạy Backward Stepwise (có thể mất 1-2 phút)...")
ols_step, step_cols = backward_stepwise_ols(X_train, y_train)
print(f"  OLS Stepwise — R² train: {ols_step.rsquared:.4f} | Giữ {len(step_cols)}/{X_train.shape[1]} biến")

# Model 3: Ridge với CV chọn alpha
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5).fit(X_train, y_train)
print(f"  Ridge       — alpha={ridge.alpha_} | R² train: {ridge.score(X_train, y_train):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 1 — TÍNH MAE, RMSE, R² TRÊN TẬP TEST
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BƯỚC 1: SO SÁNH ĐỘ ĐO SAI SỐ TRÊN TẬP TEST")
print("=" * 60)

y_pred_full  = ols_full.predict(X_test_sm)
y_pred_step  = ols_step.predict(sm.add_constant(X_test[:, step_cols]))
y_pred_ridge = ridge.predict(X_test)

def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

results = [
    compute_metrics(y_test, y_pred_full,  "OLS Full Model"),
    compute_metrics(y_test, y_pred_step,  "OLS Stepwise"),
    compute_metrics(y_test, y_pred_ridge, "Ridge Regression"),
]
print("\n  Bảng tổng hợp:")
print(pd.DataFrame(results).to_string(index=False))

# Xác định mô hình tốt nhất để dùng cho Residual Analysis
best_idx  = int(np.argmax([r['R2'] for r in results]))
best_name = results[best_idx]['Model']
best_y_pred_train = [
    ols_full.fittedvalues,
    ols_step.fittedvalues,
    ridge.predict(X_train)
][best_idx]
print(f"\n  → Mô hình tốt nhất: {best_name} — dùng cho Residual Analysis")


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 2 — 4 BIỂU ĐỒ PHÂN TÍCH PHẦN DƯ (mô hình tốt nhất)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  BƯỚC 2: VẼ 4 BIỂU ĐỒ PHÂN TÍCH PHẦN DƯ ({best_name})")
print("=" * 60)

# Tính residuals từ mô hình tốt nhất
train_fitted    = best_y_pred_train
train_resid     = y_train - train_fitted
train_std_resid = train_resid / (train_resid.std() + 1e-8)
train_sqrt_std  = np.sqrt(np.abs(train_std_resid))

# Leverage và Cook's Distance: tính từ OLS Full (Hat Matrix)
# (statsmodels OLSInfluence chỉ hỗ trợ OLS — dùng để tính leverage)
influence      = OLSInfluence(ols_full)
train_leverage = influence.hat_matrix_diag
train_cooks_d  = influence.cooks_distance[0]

fig = plt.figure(figsize=(14, 11))
fig.suptitle(f"Phân tích Phần dư — {best_name}\n(Video Games Sales)",
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

# Plot 1: Residuals vs Fitted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(train_fitted, train_resid, alpha=0.2, s=8, color='steelblue', edgecolors='none')
ax1.axhline(0, color='red', linestyle='--', linewidth=1.2)
if HAS_LOWESS:
    sm_ = lowess(train_resid, train_fitted, frac=0.3)
    ax1.plot(sm_[:, 0], sm_[:, 1], color='orange', linewidth=1.8)
ax1.set(xlabel="Fitted values", ylabel="Residuals",
        title="1. Residuals vs Fitted\n→ Kiểm tra tính tuyến tính")

# Plot 2: Normal Q-Q Plot
ax2 = fig.add_subplot(gs[0, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(train_std_resid, dist="norm")
ax2.plot(osm, osr, 'o', alpha=0.25, markersize=3, color='steelblue')
ax2.plot(osm, slope * np.array(osm) + intercept, 'r--', linewidth=1.5)
ax2.set(xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
        title="2. Normal Q-Q Plot\n→ Kiểm tra phân phối chuẩn")

# Plot 3: Scale-Location
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(train_fitted, train_sqrt_std, alpha=0.2, s=8, color='steelblue', edgecolors='none')
if HAS_LOWESS:
    sm3 = lowess(train_sqrt_std, train_fitted, frac=0.3)
    ax3.plot(sm3[:, 0], sm3[:, 1], color='orange', linewidth=1.8)
ax3.set(xlabel="Fitted values", ylabel="√|Std Residuals|",
        title="3. Scale-Location\n→ Kiểm tra phương sai đều")

# Plot 4: Leverage vs Residuals (Cook's Distance)
ax4 = fig.add_subplot(gs[1, 1])
sc = ax4.scatter(train_leverage, train_std_resid, c=train_cooks_d,
                 cmap='YlOrRd', alpha=0.4, s=8, edgecolors='none')
plt.colorbar(sc, ax=ax4, label="Cook's Distance")
thresh = 4 / len(train_resid)
mask   = train_cooks_d > thresh
ax4.scatter(train_leverage[mask], train_std_resid[mask],
            color='red', s=25, zorder=5, label=f"Outliers ({mask.sum()})")
ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax4.set(xlabel="Leverage", ylabel="Std Residuals",
        title="4. Leverage vs Residuals\n→ Tìm điểm ảnh hưởng lớn")
ax4.legend(fontsize=8)

save_report_figure('residual_analysis_part2.png', dpi=150,
                   bbox_inches='tight', facecolor='white')
print("  ✓ Đã lưu: residual_analysis_part2.png")
plt.close(fig)

_, shapiro_p = stats.shapiro(train_std_resid[:200])
print(f"  Shapiro-Wilk p = {shapiro_p:.4f} | Outliers (Cook's D > {thresh:.4f}): {mask.sum()}")


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 3 — FEATURE IMPORTANCE (dùng Ridge coef — ổn định hơn OLS)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BƯỚC 3: VẼ FEATURE IMPORTANCE (Top 20 — Ridge coef)")
print("=" * 60)

coefs      = ridge.coef_          # Ridge đã regularization → không bị inflate
importance = np.abs(coefs)
top_idx    = np.argsort(importance)[-20:]
top_names  = [feature_names_encoded[i] for i in top_idx]
top_vals   = importance[top_idx]

fig2, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, 20))
bars = ax.barh(top_names, top_vals, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, top_vals):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
ax.set(xlabel="Độ lớn hệ số Ridge chuẩn hóa |β|",
       title="Top 20 Feature Importance — Ridge Regression\n(dùng Ridge coef để tránh inflate do multicollinearity)")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_report_figure('feature_importance.png', dpi=150,
                   bbox_inches='tight', facecolor='white')
print("  ✓ Đã lưu: feature_importance.png")
plt.close(fig2)

print("\n  Top 10 features quan trọng nhất:")
fi_df = pd.DataFrame({
    'Feature': [feature_names_encoded[i] for i in np.argsort(importance)[::-1][:10]],
    '|Ridge coef|': importance[np.argsort(importance)[::-1][:10]]
}).reset_index(drop=True)
print(fi_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# BƯỚC 4 — KERNEL RIDGE REGRESSION (đúng công thức đề — +0.5 điểm)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BƯỚC 4: KERNEL RIDGE REGRESSION — +0.5 điểm thưởng")
print("=" * 60)
print("  Công thức: ŷ(x) = k(x)ᵀ (K + λI)⁻¹ y")
print("  Kernel RBF: k(x,x') = exp(-||x-x'||² / 2ℓ²)")

class KernelRidgeRegression:
    """
    Kernel Ridge Regression — tự cài đặt thuần NumPy.
    Đúng theo công thức đề bài mục 2.4:
      ŷ(x) = k(x)ᵀ (K + λI)⁻¹ y
    với K[i,j] = k(x_i, x_j) là ma trận Gram,
    k(x,x') = exp(-||x-x'||² / 2ℓ²) là kernel RBF.

    Tham số:
      lam          : hệ số regularization λ (tương tự Ridge)
      length_scale : ℓ kiểm soát độ rộng của kernel
    """
    def __init__(self, lam=1.0, length_scale=1.0):
        self.lam = lam
        self.l   = length_scale

    def _rbf_matrix(self, X1, X2):
        """Tính ma trận Gram K[i,j] = k(X1[i], X2[j])."""
        diff = X1[:, None, :] - X2[None, :, :]        # (n1, n2, p)
        sq_dist = np.sum(diff ** 2, axis=2)            # (n1, n2)
        return np.exp(-sq_dist / (2 * self.l ** 2))

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        n = len(self.y_train)
        K = self._rbf_matrix(self.X_train, self.X_train)  # (n, n)
        # Giải hệ (K + λI) α = y  →  α = (K + λI)⁻¹ y
        self.alpha = np.linalg.solve(K + self.lam * np.eye(n), self.y_train)
        return self

    def predict(self, X_test):
        # k(x)ᵀ α  với k(x)[i] = k(x, x_i)
        K_test = self._rbf_matrix(np.array(X_test, dtype=float), self.X_train)
        return K_test @ self.alpha


# Dùng tập con và chỉ numerical features (tránh curse of dimensionality)
N_TR, N_TE = 300, 100
num_idx  = [feature_names_encoded.index(c)
            for c in ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']]
X_kr_tr  = X_train[:N_TR][:, num_idx]
X_kr_te  = X_test[:N_TE][:, num_idx]
y_kr_tr  = y_train[:N_TR]
y_kr_te  = y_test[:N_TE]

print(f"\n  Dùng {N_TR} train / {N_TE} test | {len(num_idx)} numerical features")
print("  Đang chọn λ và ℓ qua 3-fold CV...")

kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_lam, best_ls, best_cv = 1.0, 1.0, np.inf

for lam in [0.1, 1.0, 10.0]:
    for ls in [0.5, 1.0, 2.0]:
        rmses = []
        for tr_i, vl_i in kf.split(X_kr_tr):
            kr = KernelRidgeRegression(lam=lam, length_scale=ls)
            kr.fit(X_kr_tr[tr_i], y_kr_tr[tr_i])
            rmses.append(np.sqrt(mean_squared_error(
                y_kr_tr[vl_i], kr.predict(X_kr_tr[vl_i]))))
        avg = np.mean(rmses)
        print(f"    λ={lam:<5}  ℓ={ls}  →  CV RMSE={avg:.4f}")
        if avg < best_cv:
            best_cv, best_lam, best_ls = avg, lam, ls

print(f"\n  → Tốt nhất: λ={best_lam}, ℓ={best_ls}")

kr_final  = KernelRidgeRegression(lam=best_lam, length_scale=best_ls)
kr_final.fit(X_kr_tr, y_kr_tr)
y_pred_kr = kr_final.predict(X_kr_te)
mae_kr    = mean_absolute_error(y_kr_te, y_pred_kr)
rmse_kr   = np.sqrt(mean_squared_error(y_kr_te, y_pred_kr))
r2_kr     = r2_score(y_kr_te, y_pred_kr)
print(f"  Kernel Ridge  MAE={mae_kr:.4f}  RMSE={rmse_kr:.4f}  R²={r2_kr:.4f}")


# ── Vẽ so sánh Kernel Ridge vs OLS trên chiều Critic_Score ──────────
cs_local_idx = 0   # Critic_Score là cột đầu tiên trong num_idx
X_1d = X_kr_tr[:, cs_local_idx].reshape(-1, 1)

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Kernel Ridge Regression vs OLS — minh họa trên Critic_Score (data thật)",
              fontsize=12, fontweight='bold')

# Plot trái: ảnh hưởng của length_scale ℓ
for ls_p, col, lbl in [(0.3, 'tomato',    'ℓ=0.3 (overfit)'),
                        (1.0, 'steelblue', 'ℓ=1.0'),
                        (5.0, 'green',     'ℓ=5.0 (underfit)')]:
    kr1 = KernelRidgeRegression(lam=best_lam, length_scale=ls_p).fit(X_1d, y_kr_tr)
    xl  = np.linspace(X_1d.min(), X_1d.max(), 80).reshape(-1, 1)
    axes[0].plot(xl, kr1.predict(xl), color=col, lw=2, label=lbl)
axes[0].scatter(X_1d, y_kr_tr, alpha=0.25, s=10, color='gray')
axes[0].set(xlabel="Critic_Score (chuẩn hóa)", ylabel="Global_Sales",
            title="Ảnh hưởng của length_scale ℓ\n(tương tự bandwidth trong Nadaraya-Watson)")
axes[0].legend(fontsize=8)
axes[0].spines[['top', 'right']].set_visible(False)

# Plot phải: OLS vs Kernel Ridge tốt nhất
ols1d  = sm.OLS(y_kr_tr, sm.add_constant(X_1d)).fit()
xl     = np.linspace(X_1d.min(), X_1d.max(), 80).reshape(-1, 1)
krbest = KernelRidgeRegression(lam=best_lam, length_scale=best_ls).fit(X_1d, y_kr_tr)

axes[1].scatter(X_1d, y_kr_tr, alpha=0.25, s=10, color='gray', label='Data')
axes[1].plot(xl, ols1d.predict(sm.add_constant(xl)),
             'tomato', lw=2, label='OLS (tuyến tính)')
axes[1].plot(xl, krbest.predict(xl),
             'steelblue', lw=2, label=f'Kernel Ridge (λ={best_lam}, ℓ={best_ls})')
axes[1].set(xlabel="Critic_Score (chuẩn hóa)", ylabel="Global_Sales",
            title="OLS vs Kernel Ridge Regression\n(Kernel bắt được phi tuyến cục bộ)")
axes[1].legend(fontsize=8)
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_report_figure('kernel_regression_comparison.png', dpi=150,
                   bbox_inches='tight', facecolor='white')
print("  ✓ Đã lưu: kernel_regression_comparison.png")
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════
# TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TỔNG KẾT SO SÁNH CÁC MÔ HÌNH")
print("=" * 60)

summary = pd.DataFrame(results + [
    {
        'Model': f'Kernel Ridge (λ={best_lam}, ℓ={best_ls})',
        'MAE':  mae_kr,
        'RMSE': rmse_kr,
        'R2':   r2_kr
    }
])
print(summary.to_string(index=False))
best_overall = summary.loc[summary['R2'].idxmax(), 'Model']
print(f"\n  → Mô hình tốt nhất theo R²: {best_overall}")
print(f"  → Residual Analysis đã dùng: {best_name}")
print(f"\n  Files đã tạo (report/images/):")
print("    residual_analysis_part2.png")
print("    feature_importance.png")
print("    kernel_regression_comparison.png")
print("\n  XONG! ✓")
