# Đồ Án 2: Data Fitting và Phương Pháp OLS

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
