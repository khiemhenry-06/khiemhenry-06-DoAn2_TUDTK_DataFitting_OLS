# Đồ Án 2: Data Fitting và Phương Pháp OLS

> [!IMPORTANT]
> ## 📊 BẢNG THEO DÕI TIẾN ĐỘ THỰC HIỆN ĐỒ ÁN 2 (CẬP NHẬT MỚI NHẤT)
> Bảng này giúp nhóm trưởng và các thành viên dễ dàng kiểm soát các đầu việc đã hoàn thành và đốc thúc các thành viên còn lại nộp bài đúng hạn.
> 
> * **Tổng số thành viên:** 5 thành viên (Khiêm, Nguyên, Minh, Nam, Kiên)
> * **Số nhiệm vụ đã hoàn thành:** 4/7 nhiệm vụ chính (~57%)
> * **Trạng thái nhánh dự án:** Đã đồng bộ hóa đồng nhất dữ liệu EDA (Nam) -> Pipeline (Khiêm) -> Mô hình (Minh) vào file Jupyter Notebook chính thức (`part2_notebook.ipynb`).
> 
> ### 1. PHẦN 1: BÁO CÁO LÝ THUYẾT & XÂY DỰNG OLS TỪ ĐẦU (FROM SCRATCH)
> 
> | Thành viên | Nhiệm vụ đảm nhận | File Code liên quan | Trạng thái | Ghi chú & Đánh giá |
> | :--- | :--- | :--- | :---: | :--- |
> | **Nguyên** | Xây dựng thuật toán hồi quy OLS từ đầu bằng toán ma trận NumPy | `part1/ols_implementation.py` | 🟢 **Hoàn thành** | Chứa đầy đủ tính toán hệ số $\hat{\beta}$, các chỉ số thống kê $R^2$, Adj-$R^2$, kiểm định $t$, kiểm định $F$, và kiểm tra đa cộng tuyến. Code chạy rất chuẩn xác! |
> | **Nam** | Viết thuật toán và phân tích lý thuyết về Ridge, Lasso, và Cross Validation | `part1/ridge_lasso.py`<br>`part1/cross_validation.py` | 🟡 **Chưa làm** *(Đang thực hiện)* | Hiện tại các file code chỉ chứa các dòng comment trống phân chia nhiệm vụ. |
> | **Kiên** | Xây dựng lý thuyết và vẽ biểu đồ phân tích phần dư (Residual Analysis) | `part1/residual_analysis.py` | 🟡 **Chưa làm** *(Đang thực hiện)* | Hiện tại file code chỉ chứa các dòng comment trống. |
> 
> ### 2. PHẦN 2: ỨNG DỤNG HỒI QUY TRÊN DỮ LIỆU THỰC TẾ (VIDEO GAMES SALES)
> 
> | Thành viên | Nhiệm vụ đảm nhận | File Code liên quan | Trạng thái | Ghi chú & Đánh giá |
> | :--- | :--- | :--- | :---: | :--- |
> | **Nam** | Phân tích dữ liệu khám phá (EDA), vẽ các biểu đồ tương quan, Boxplot, Scatterplot | `part2/part2_eda_vgsales.py`<br>*(Đã đồng bộ sang Cell 3 của Notebook)* | 🟢 **Hoàn thành** | Đã hoàn thành 3 biểu đồ chính (Heatmap Pearson, Boxplot phân phối loại bỏ outliers $>20M$, Scatterplot Critic Score). |
> | **Khiêm** | Thiết kế Pipeline tiền xử lý dữ liệu tùy biến (`DataPipeline` kế thừa Scikit-Learn) | `part2/data_pipeline.py`<br>*(Đã đồng bộ sang Cell 5 của Notebook)* | 🟢 **Hoàn thành** | Pipeline xử lý cực kỳ chuyên nghiệp (imputer dữ liệu khuyết, mã hóa OneHot, scale Standard). Ngăn chặn rò rỉ dữ liệu chuẩn xác. |
> | **Minh** | Huấn luyện 3 mô hình hồi quy (OLS Full, Stepwise Backward, Ridge CV) và xuất dự đoán | `part2/model_comparison.py`<br>*(Đã đồng bộ sang Cell 8, 10, 12, 14 Notebook)* | 🟢 **Hoàn thành** | Đã huấn luyện thành công cả 3 mô hình, tìm ra $\alpha$ tối ưu bằng Cross-Validation, và xuất thành công các file dự đoán ra thư mục `data/`. |
> | **Kiên** | Đánh giá sai số (RMSE, MAE), phân tích phần dư thực tế trên dữ liệu dự báo | `part2/advanced_methods.py`<br>*(Đã đồng bộ sang Cell 15, 16 Notebook)* | 🟡 **Chưa làm** *(Đang thực hiện)* | Hiện tại file code chỉ chứa các dòng comment trống chờ kết quả dự báo từ nhóm. |
> 
> ---
> 
> ### 📈 KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO (ACTION ITEMS)
> - [ ] **Trần Thanh Nam:** Hoàn thiện lý thuyết & code phần Ridge/Lasso và Cross Validation ở Phần 1 (`part1/ridge_lasso.py`, `part1/cross_validation.py`).
> - [ ] **Đỗ Trung Kiên:** 
>     1. Hoàn thiện phần lý thuyết Residual Analysis ở Phần 1 (`part1/residual_analysis.py`).
>     2. Sử dụng file kết quả dự đoán `data/model_predictions.csv` đã có sẵn để viết tiếp code đánh giá sai số & vẽ đồ thị phân phối phần dư ở cuối Notebook (`part2_notebook.ipynb`).
> - [ ] **Tiến hành biên soạn báo cáo LaTeX:** Sau khi các thành viên nộp đủ code, nhóm trưởng Khiêm tiến hành ráp kết quả vào file báo cáo LaTeX trong thư mục `report/report.tex`.

---

## Thông Tin Nhóm

| Thành viên | MSSV | Vai trò |
|------------|------|---------|
| Lê Phạm Đăng Khiêm | 24120341 | Quản lý dự án & Tiền xử lý dữ liệu |
| Nguyễn Công Nguyên | 24120106 | Kỹ sư Thuật toán – Phần 1 |
| Trần Thanh Nam | 24120099 | Mô phỏng & Khảo sát – Phần 1 & 2 |
| Lê Quang Minh | 24120092 | Chuyên gia Mô hình – Phần 2 |
| Đỗ Trung Kiên | 24120350 | Đánh giá & Kỹ thuật nâng cao |


**Môn học:** Toán Ứng Dụng và Thống Kê – FIT HCMUS

---

## Hướng Dẫn Làm Việc Nhóm Trên GitHub

Để đảm bảo tiến độ và tránh xung đột mã nguồn (conflict), nhóm thống nhất quy trình sau:

### 1. Quản lý Nhánh (Branching Strategy)
- **`main`**: Nhánh chính, chứa code ổn định. Không push code trực tiếp lên nhánh này trừ các file tài liệu cơ bản ban đầu.
- **Nhánh tính năng**: Mỗi thành viên tạo nhánh riêng từ `main` để làm nhiệm vụ của mình theo cú pháp `feature/<ten-thanh-vien>-<ten-tinh-nang>`.
  - Ví dụ: `feature/khiem-datapipeline`, `feature/nguyen-ols`, `feature/nam-eda`.

### 2. Quy trình làm việc hàng ngày
1. Cập nhật code mới nhất từ repo về máy:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Tạo nhánh mới để bắt đầu code (nếu chưa tạo):
   ```bash
   git checkout -b feature/ten-nhanh
   ```
3. Trong quá trình code, commit thường xuyên kèm thông điệp rõ nghĩa:
   ```bash
   git add .
   git commit -m "feat: hoàn thành xử lý missing values cho DataPipeline"
   ```
4. Đẩy code lên GitHub định kỳ:
   ```bash
   git push origin feature/ten-nhanh
   ```

### 3. Pull Request (PR) & Review
- Khi đã làm xong, lên GitHub chọn **New Pull Request** để merge từ nhánh của bạn vào `main`.
- Gán tag (Assignee) tên mình và add Reviewer là Khiêm (PM) hoặc các bạn code cùng phần đó.
- Sau khi đã review ok, tiến hành **Merge Pull Request**.

### 4. Một số quy tắc chung
- **Tuyệt đối không commit** các file môi trường ảo (`venv/`), cache (`__pycache__/`) hay checkpoint của notebook.
- **Hạn chế xung đột (Conflict)**: Liên tục pull code mới nhất về gộp vào nhánh feature của mình trước khi tạo PR.
- Dữ liệu lớn (`.csv`) chỉ cần push một lần vào đúng thư mục `part2/data/`, không cần push lại nhiều lần.

---

## Mô Tả Đồ Án

Đồ án gồm 2 phần chính:

1. **Phần 1 – Lý thuyết & Minh họa:** Cài đặt thủ công thuật toán OLS, ma trận Hat, phân tích phần dư, mô phỏng kiểm chứng Gauss-Markov bằng Python (không dùng sklearn cho luồng chính).

2. **Phần 2 – Ứng dụng thực tế:** Áp dụng hồi quy tuyến tính trên bộ dữ liệu **Video Games Sales** để dự báo doanh thu toàn cầu (`Global_Sales`), bao gồm tiền xử lý dữ liệu, xây dựng mô hình, và đánh giá kết quả.

---

## Bộ Dữ Liệu

- **Tên:** Video Game Sales with Ratings
- **Nguồn:** [Kaggle – rush4ratio/video-game-sales-with-ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
- **Kích thước:** 16,719 dòng × 16 cột
- **Biến mục tiêu:** `Global_Sales` (doanh thu toàn cầu, triệu USD)
- **Missing values:** Critic_Score (51.3%), User_Score (40.1% + 2,425 giá trị "tbd"), User_Count (54.6%), Developer (39.6%), Rating (40.5%)

File dữ liệu gốc: `part2/data/video_games_sales.csv`

---

## Cấu Trúc Thư Mục

```
.
├── README.md                         # File hướng dẫn (file này)
├── requirements.txt                  # Các thư viện Python cần cài đặt
├── report/
│   ├── report.tex                    # Báo cáo LaTeX
│   └── report.pdf                    # Báo cáo đã biên dịch
├── part1/                            # Phần 1: Lý thuyết và minh họa
│   ├── ols_implementation.py         # Cài đặt OLS từ đầu
│   ├── ridge_lasso.py                # Ridge & Lasso Regression
│   ├── residual_analysis.py          # Phân tích phần dư
│   ├── cross_validation.py           # K-fold Cross Validation
│   └── part1_notebook.ipynb          # Notebook minh họa lý thuyết
└── part2/                            # Phần 2: Ứng dụng thực tế
    ├── data/
    │   └── video_games_sales.csv     # Dữ liệu gốc
    ├── data_pipeline.py              # Pipeline tiền xử lý dữ liệu
    ├── model_comparison.py           # So sánh mô hình
    ├── advanced_methods.py           # Kỹ thuật nâng cao (Kernel/Bayesian)
    └── part2_notebook.ipynb          # Notebook phân tích và thảo luận
```

---

## Hướng Dẫn Cài Đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd Group_<ID>
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Chạy Notebook

```bash
jupyter notebook
```

Mở lần lượt:
- `part1/part1_notebook.ipynb` — Phần lý thuyết
- `part2/part2_notebook.ipynb` — Phần ứng dụng

---

## Hướng Dẫn Sử Dụng Data Pipeline

Class `DataPipeline` (tại `part2/data_pipeline.py`) đã được hoàn thiện giúp tự động hóa toàn bộ quá trình làm sạch và định dạng lại dữ liệu. Mọi người chỉ cần import vào file code/notebook của mình để nhận dữ liệu sạch đưa thẳng vào mô hình:

### 1. Import và Khởi tạo
```python
import pandas as pd
from data_pipeline import DataPipeline

# Đọc dữ liệu thô
df = pd.read_csv("data/video_games_sales.csv")

# Khởi tạo pipeline (mặc định giữ top 30 nhà phát hành để tránh quá nhiều cột thưa)
pipeline = DataPipeline(top_n_publishers=30)
```

### 2. Xử lý không bị rò rỉ dữ liệu (Data Leakage Prevention)
Quy trình chuẩn bắt buộc phải chia tập dữ liệu thô trước, sau đó thực hiện học các chỉ số (`fit`) trên tập `Train` rồi mới áp dụng sang tập `Test`:

```python
# Giả sử đã chia X_train, X_test thô từ dataset gốc...

# Bước 1: "Fit" và xử lý tập Train (Tính median, mean, std và học bộ cột dummy)
X_train_clean = pipeline.fit_transform(X_train)

# Bước 2: Chỉ "Transform" trên tập Test (Dùng lại thông số của tập Train)
X_test_clean = pipeline.transform(X_test)
```

### 3. Các tính năng đã được tự động hóa bên trong:
*   **Xử lý "tbd"**: Tự tìm chuỗi `tbd` ở `User_Score` -> ép về `NaN` -> chuyển kiểu dữ liệu `float`.
*   **Điền khuyết (Imputation)**: Tự điền khuyết biến số bằng `Median` tính từ tập Train.
*   **Mã hóa & Rút gọn**: Gom nhóm các `Publisher` nhỏ lẻ thành nhóm `"Other"`, sau đó tự động mã hóa One-Hot Encoding cho `Platform`, `Genre`, `Publisher`.
*   **Căn chỉnh chiều**: Tự sinh thêm cột giả (`0`) hoặc cắt cột thừa ở tập `Test` sao cho **số lượng và thứ tự cột ở tập Train và Test giống hệt nhau 100%** (đảm bảo nhân ma trận $(X^TX)^{-1}X^Ty$ không bị lỗi).
*   **Chuẩn hóa**: Áp dụng chuẩn hóa Z-score cho mọi thuộc tính số.

---

## Reproducibility


Tất cả kết quả có thể tái lập được. Seed mặc định: `random_state=42`.

---

## Phân Công Công Việc

| Giai đoạn | Thành viên | Nhiệm vụ | Hạn chót |
|-----------|-----------|-----------|---------|
| 1 (12-17/05) | Khiêm | Cấu trúc thư mục, DataPipeline | 17/05 |
| 1 (12-17/05) | Nguyên | OLS, Hat matrix, kiểm định | 17/05 |
| 1 (12-15/05) | Nam | EDA bộ dữ liệu Video Games | 15/05 |
| 2 (18-22/05) | Nam | Monte Carlo, K-fold CV | 20/05 |
| 2 (18-22/05) | Minh | Train/Test split, 3 mô hình | 22/05 |
| 3 (23-26/05) | Kiên & Nguyên | Đánh giá, Kỹ thuật nâng cao | 26/05 |
| 4 (27-29/05) | Khiêm & Minh | Tổng hợp báo cáo, rà soát | 28/05 |
