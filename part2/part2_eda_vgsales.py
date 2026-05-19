import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Setup style vẽ biểu đồ cho đẹp và đồng nhất
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8) # Kích thước mặc định cho các biểu đồ

# --- 1. LOAD DỮ LIỆU ---
print("Đang load dữ liệu...")
# Cơ chế tìm đường dẫn thông minh để tránh lỗi FileNotFoundError
csv_path = 'data/video_games_sales.csv'
if not os.path.exists(csv_path):
    csv_path = 'part2/data/video_games_sales.csv'
df = pd.read_csv(csv_path)

# --- 2. TIỀN XỬ LÝ DỮ LIỆU SƠ BỘ ---
print("Đang tiền xử lý sơ bộ...")
# Ép kiểu User_Score, chuyển 'tbd' và các lỗi khác thành NaN để tính toán được
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# --- 3. KHẢO SÁT DỮ LIỆU (EDA) ---

# BIỂU ĐỒ 1: HEATMAP - Ma trận tương quan Pearson
# Mục tiêu: Đánh giá mối quan hệ tuyến tính sơ bộ và phát hiện Data Leakage.
print("\n--- Đang vẽ Biểu đồ 1: Heatmap tương quan ---")
# Lọc chỉ các biến số
numeric_df = df.select_dtypes(include=[np.number])

# Tóm tắt nhanh dữ liệu khuyết
missing_numeric = numeric_df.isnull().sum() / len(numeric_df) * 100
print("Tỷ lệ dữ liệu khuyết (%) ở các biến số:\n", missing_numeric[missing_numeric > 0].sort_values(ascending=False))

# Tính ma trận tương quan
corr_matrix = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5) # fmt=".2f" để hiển thị 2 chữ số thập phân
plt.title("Ma trận tương quan Pearson giữa các biến số")
plt.tight_layout()

# BIỂU ĐỒ 2: BOXPLOT - Phân phối Global_Sales theo Genre
# Mục tiêu: Phát hiện Outliers (các game bom tấn) và so sánh doanh thu các thể loại.
print("\n--- Đang vẽ Biểu đồ 2: Boxplot Global_Sales theo Genre ---")

# Lọc bỏ các game có doanh thu quá cực đoan (>20 triệu USD) để boxplot phổ biến dễ nhìn hơn
# nhưng vẫn giữ lại được các outliers có ý nghĩa thống kê.
subset_df = df[df['Global_Sales'] < 20]

plt.figure(figsize=(16, 9))
sns.boxplot(x='Genre', y='Global_Sales', data=subset_df, order=subset_df['Genre'].value_counts().index)
plt.title("Phân phối doanh thu (Global_Sales) theo Thể loại (Genre)\n(Giới hạn doanh thu < 20 triệu USD để dễ quan sát)")
plt.xticks(rotation=45)
plt.ylabel("Doanh thu toàn cầu (Triệu USD)")
plt.tight_layout()

# BIỂU ĐỒ 3: SCATTER PLOT - Mối quan hệ giữa Critic_Score và Global_Sales
# Mục tiêu: Trực quan hóa mối quan hệ tuyến tính mà Heatmap đã chỉ ra.
print("\n--- Đang vẽ Biểu đồ 3: Scatter plot Critic_Score vs Global_Sales ---")

plt.figure(figsize=(11, 7))
# Scatter plot cho điểm chuyên gia vs doanh thu
# alpha=0.5 để làm mờ điểm, dễ thấy các khu vực tập trung dày đặc
# Loại bỏ NaN ở cả 2 cột để vẽ mượt hơn
scatter_data = df[['Critic_Score', 'Global_Sales']].dropna()

sns.scatterplot(x='Critic_Score', y='Global_Sales', data=scatter_data, alpha=0.4, color='b')
plt.title("Mối quan hệ giữa Điểm chuyên gia (Critic_Score) và Doanh thu (Global_Sales)")
plt.xlabel("Điểm chuyên gia")
plt.ylabel("Doanh thu toàn cầu (Triệu USD)")
plt.tight_layout()

# Hiển thị biểu đồ ngay sau khi chạy
plt.show()

print("\n--- Đã chạy xong EDA! ---")