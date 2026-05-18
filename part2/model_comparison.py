import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import sys
import os

# --- CƠ CHẾ TỰ ĐỘNG TÌM ĐƯỜNG DẪN ---
# Lấy đường dẫn tuyệt đối của thư mục part2 (nơi chứa file model_comparison.py này)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục part2 vào hệ thống để import được data_pipeline.py nằm cùng chỗ
sys.path.append(current_dir)
from data_pipeline import DataPipeline 

print("--- BƯỚC 1: ĐỌC VÀ CHIA DỮ LIỆU ---")
# Tạo đường dẫn tuyệt đối đi thẳng tới file csv
csv_path = os.path.join(current_dir, "data", "video_games_sales.csv")
print(f"[*] Đang đọc dữ liệu từ: {csv_path}")

# Đọc file dữ liệu
df = pd.read_csv(csv_path)

# Xóa các dòng bị thiếu biến mục tiêu (Global_Sales)
df = df.dropna(subset=['Global_Sales'])

# Tách biến đặc trưng (X) và biến mục tiêu (y)
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

# Chia tập Train/Test với tỉ lệ 80:20
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Kích thước tập Train thô: {X_train_raw.shape}")
print(f"Kích thước tập Test thô: {X_test_raw.shape}")


print("\n--- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ---")
# Khởi tạo Pipeline (giữ top 30 nhà phát hành)
pipeline = DataPipeline(top_n_publishers=30)

# Fit (học) và Transform (biến đổi) trên tập Train
X_train_clean = pipeline.fit_transform(X_train_raw)

# Chỉ Transform (biến đổi) trên tập Test 
X_test_clean = pipeline.transform(X_test_raw)

print(f"Kích thước tập Train sau xử lý: {X_train_clean.shape}")
print(f"Kích thước tập Test sau xử lý: {X_test_clean.shape}")


print("\n--- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH 1 - OLS CƠ BẢN (FULL MODEL) ---")
# 1. ÉP KIỂU VỀ SỐ THỰC (FLOAT) ĐỂ TRÁNH LỖI OBJECT CỦA STATSMODELS
X_train_clean = X_train_clean.astype(float)
X_test_clean = X_test_clean.astype(float)
y_train = y_train.astype(float)

# 2. Thêm cột hằng số (Intercept) cho statsmodels
X_train_sm = sm.add_constant(X_train_clean)
X_test_sm = sm.add_constant(X_test_clean)

# Khởi tạo và huấn luyện mô hình OLS
full_model = sm.OLS(y_train, X_train_sm).fit()

# In bảng tóm tắt kết quả
print(full_model.summary())

from sklearn.linear_model import RidgeCV

print("\n" + "="*60)
print("--- BƯỚC 4: HUẤN LUYỆN MÔ HÌNH 2 - OLS CHỌN BIẾN (STEPWISE BACKWARD) ---")
print("="*60)

def backward_elimination(X, y, significance_level=0.05):
    """
    Hàm tự động loại bỏ dần các biến có p-value > 0.05
    """
    features = X.columns.tolist()
    while len(features) > 0:
        # Thêm hằng số vào mô hình
        X_with_constant = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_constant).fit()
        
        # Lấy p-values, bỏ qua intercept (const) để không loại bỏ nó
        p_values = model.pvalues.drop('const', errors='ignore')
        max_p_value = p_values.max()
        
        # Nếu p-value lớn nhất vẫn vượt ngưỡng 0.05 thì loại bỏ biến đó
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            print(f"Đã loại bỏ: {excluded_feature: <30} (p-value = {max_p_value:.4f})")
        else:
            break  # Dừng vòng lặp khi tất cả biến đều có ý nghĩa thống kê
            
    # Huấn luyện lại mô hình cuối cùng với các biến đã chọn
    final_model = sm.OLS(y, sm.add_constant(X[features])).fit()
    return final_model, features

# Chạy thuật toán chọn biến
short_model, selected_features = backward_elimination(X_train_clean, y_train)

print("\n--- KẾT QUẢ MÔ HÌNH 2 (SHORT MODEL) ---")
print(f"Số lượng biến giữ lại: {len(selected_features)} / {X_train_clean.shape[1]}")
# In bảng tóm tắt của mô hình rút gọn
print(short_model.summary())


print("\n" + "="*60)
print("--- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH 3 - RIDGE REGRESSION ---")
print("="*60)

# Định nghĩa các giá trị siêu tham số alpha (lambda) cần thử nghiệm
alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Khởi tạo mô hình Ridge với Cross-Validation (k-fold = 5)
# Thư viện sẽ tự động chia 5 phần và tìm ra alpha tốt nhất
ridge_cv_model = RidgeCV(alphas=alphas_to_test, cv=5)

# Huấn luyện trên toàn bộ tập đặc trưng đã qua tiền xử lý
ridge_cv_model.fit(X_train_clean, y_train)

print(f"Giá trị alpha (lambda) tốt nhất được chọn qua CV: {ridge_cv_model.alpha_}")
print(f"R^2 score trên tập Train: {ridge_cv_model.score(X_train_clean, y_train):.4f}")

print("\n--- BƯỚC 6: XUẤT KẾT QUẢ CHO PHẦN ĐÁNH GIÁ ---")
# Sử dụng current_dir đã được lấy tự động ở đầu file để ráp đường dẫn
features_path = os.path.join(current_dir, "data", "selected_features.csv")
predictions_path = os.path.join(current_dir, "data", "model_predictions.csv")

# Lưu danh sách các đặc trưng đã chọn từ Mô hình 2
pd.Series(selected_features).to_csv(features_path, index=False)
print(f"[*] Đã xuất danh sách biến chọn lọc ra: {features_path}")

# Tạo bảng chứa giá trị thực tế (y_test) và giá trị dự đoán từ 3 mô hình
results = pd.DataFrame({'Actual_Sales': y_test})

# Dự đoán từ Mô hình 1 (OLS Cơ bản)
results['Pred_Full_Model'] = full_model.predict(X_test_sm)

# Dự đoán từ Mô hình 2 (OLS Chọn biến)
X_test_short = sm.add_constant(X_test_clean[selected_features])
results['Pred_Short_Model'] = short_model.predict(X_test_short)

# Dự đoán từ Mô hình 3 (Ridge CV)
results['Pred_Ridge'] = ridge_cv_model.predict(X_test_clean)

# Lưu kết quả dự đoán
results.to_csv(predictions_path, index=False)
print(f"[*] Đã xuất kết quả dự đoán ra: {predictions_path}")