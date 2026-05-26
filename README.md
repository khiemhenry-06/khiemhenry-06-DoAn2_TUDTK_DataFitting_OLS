# Đồ án 2: Data Fitting và Phương pháp OLS

Đồ án thực hiện hai nội dung chính: trình bày và cài đặt các thành phần nền tảng của Ordinary Least Squares (OLS), sau đó áp dụng hồi quy tuyến tính lên bộ dữ liệu thực tế `Video Games Sales` để dự đoán `Global_Sales`.

## Bộ dữ liệu

- **Tên dữ liệu:** Video Games Sales with Ratings
- **Nguồn:** [Kaggle - Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
- **File dữ liệu:** `part2/data/video_games_sales.csv`
- **Kích thước:** 16,719 quan sát và 16 cột
- **Biến mục tiêu:** `Global_Sales`, biểu diễn doanh thu toàn cầu của game
- **Đặc điểm chính:** dữ liệu có nhiều giá trị khuyết ở các cột điểm đánh giá như `Critic_Score`, `User_Score`, `Critic_Count`, `User_Count`, phù hợp với yêu cầu xử lý missing values của đồ án.

## Thành viên nhóm

| Thành viên | MSSV | Vai trò chính |
| --- | --- | --- |
| Lê Phạm Đăng Khiêm | 24120341 | Quản lý dự án, tiền xử lý dữ liệu, tổng hợp báo cáo |
| Lê Quang Minh | 24120092 | Huấn luyện và so sánh mô hình |
| Trần Thanh Nam | 24120099 | EDA, mô phỏng Monte Carlo, Ridge/K-Fold |
| Nguyễn Công Nguyên | 24120106 | OLS từ đầu, Hat Matrix, kiểm định, VIF |
| Đỗ Trung Kiên | 24120350 | Phân tích phần dư, feature importance, Kernel Ridge |

## Cấu trúc thư mục

```text
.
|-- README.md
|-- requirements.txt
|-- report/
|   |-- report.pdf
|   `-- report.tex
|-- part1/
|   |-- ols_implementation.py      # OLS from scratch
|   |-- ridge_lasso.py             # Ridge Regression và ridge trace
|   |-- residual_analysis.py       # Phân tích phần dư
|   |-- cross_validation.py        # K-fold Cross Validation
|   `-- part1_notebook.ipynb       # Minh họa lý thuyết
`-- part2/
    |-- data/
    |   `-- video_games_sales.csv  # Dữ liệu gốc
    |-- data_pipeline.py           # Tiền xử lý dữ liệu
    |-- model_comparison.py        # So sánh mô hình
    |-- advanced_methods.py        # Phân tích nâng cao
    `-- part2_notebook.ipynb       # Thực nghiệm phần 2
```

## Cài đặt môi trường

Yêu cầu Python 3.10+.

```bash
pip install -r requirements.txt
```

## Cách chạy và tìm hiểu đồ án

1. Đọc báo cáo chính tại `report/report.pdf`.
2. Mở `part1/part1_notebook.ipynb` để xem phần minh họa lý thuyết OLS, Ridge, K-Fold và Monte Carlo.
3. Mở `part2/part2_notebook.ipynb` để xem luồng xử lý dữ liệu thực tế, huấn luyện mô hình và đánh giá kết quả.
4. Có thể chạy từng file Python riêng:

```bash
python part1/ols_implementation.py
python part1/ridge_lasso.py
python part1/residual_analysis.py
python part1/cross_validation.py
python part2/data_pipeline.py
python part2/model_comparison.py
python part2/advanced_methods.py
```

## Kết quả và hình ảnh

Sau khi chạy notebook hoặc các script sinh biểu đồ, hình ảnh được lưu trong thư mục `report/images/` trên máy local. Thư mục này là output sinh ra khi chạy chương trình nên không được đưa lên Git theo cấu trúc nộp bài tối giản.

Các file kết quả trung gian như `part2/data/model_predictions.csv`, `part2/data/selected_features.csv`, file LaTeX phụ trợ `.aux/.log/.toc/.out`, cache Python và checkpoint notebook cũng được xem là file sinh ra trong quá trình chạy, không cần push lên repository.

## Ghi chú nộp bài

Các file cần nộp chính gồm source code trong `part1/`, `part2/`, dữ liệu gốc `video_games_sales.csv`, `requirements.txt`, `README.md`, `report/report.tex` và `report/report.pdf`.
