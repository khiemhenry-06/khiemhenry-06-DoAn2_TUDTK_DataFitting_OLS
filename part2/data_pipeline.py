# Lê Phạm Đăng Khiêm

import numpy as np
import pandas as pd
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class DataPipeline:
    def __init__(self, top_n_publishers=30):
        self.numerical_cols = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
        self.categorical_cols = ['Platform', 'Genre', 'Publisher']
        self.top_n_publishers = top_n_publishers

        # Các thống kê này chỉ được học từ tập train để tránh rò rỉ dữ liệu.
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.top_publishers = []
        self.feature_columns_ = None
        self.missing_report_ = None

    def _ensure_expected_columns(self, data):
        """Bổ sung các cột pipeline cần dùng nếu input thiếu cột."""
        for col in self.numerical_cols + self.categorical_cols:
            if col not in data.columns:
                data[col] = np.nan
        return data

    def _preprocess_basic(self, df):
        data = df.copy()
        data = self._ensure_expected_columns(data)

        # User_Score trong bộ dữ liệu gốc có giá trị 'tbd', cần xem như missing.
        data['User_Score'] = data['User_Score'].replace(['tbd', 'TBD', ''], np.nan)

        for col in self.numerical_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def fit(self, X, y=None):
        data = self._preprocess_basic(X)
        self.medians = {}
        self.means = {}
        self.stds = {}

        self.missing_report_ = (data[self.numerical_cols + self.categorical_cols]
                                .isna().mean()
                                .sort_values(ascending=False) * 100)

        # Median imputation phù hợp vì các biến điểm/đếm bị lệch và có outlier.
        for col in self.numerical_cols:
            if data[col].notna().any():
                self.medians[col] = data[col].median()
            else:
                self.medians[col] = 0.0

        if 'Publisher' in data.columns:
            publisher_series = data['Publisher'].fillna('Unknown')
            self.top_publishers = (
                publisher_series.value_counts()
                .nlargest(self.top_n_publishers)
                .index
                .tolist()
            )

        temp_data = data.copy()
        for col in self.numerical_cols:
            temp_data[col] = temp_data[col].fillna(self.medians[col])

        for col in self.numerical_cols:
            self.means[col] = temp_data[col].mean()
            std_value = temp_data[col].std()
            self.stds[col] = 1.0 if std_value == 0 or pd.isna(std_value) else std_value

        dummy_out = self._transform_internal(X)
        self.feature_columns_ = dummy_out.columns.tolist()

        return self

    def _transform_internal(self, X):
        data = self._preprocess_basic(X)

        for col in self.numerical_cols:
            fill_value = self.medians.get(col, 0.0)
            data[col] = data[col].fillna(fill_value)

        for col in self.categorical_cols:
            data[col] = data[col].fillna('Unknown')

        if len(self.top_publishers) > 0:
            data['Publisher'] = data['Publisher'].apply(
                lambda x: x if x in self.top_publishers else 'Other'
            )

        for col in self.numerical_cols:
            mean = self.means.get(col, 0.0)
            std = self.stds.get(col, 1.0)
            data[col] = (data[col] - mean) / std

        df_encoded = pd.get_dummies(
            data,
            columns=self.categorical_cols,
            prefix=self.categorical_cols,
            dtype=float
        )

        dummy_cols = [
            col for col in df_encoded.columns
            if any(col.startswith(cat + "_") for cat in self.categorical_cols)
        ]
        cols_to_keep = self.numerical_cols + dummy_cols

        return df_encoded[cols_to_keep].astype(float)

    def transform(self, X):
        if self.feature_columns_ is None:
            raise RuntimeError("Phải gọi fit() trước khi transform().")

        transformed = self._transform_internal(X)

        missing_cols = set(self.feature_columns_) - set(transformed.columns)
        for col in missing_cols:
            transformed[col] = 0.0

        return transformed.reindex(columns=self.feature_columns_, fill_value=0.0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def test_datapipeline_basic():
    print("Running test_datapipeline_basic...")
    train_data = pd.DataFrame({
        'Platform': ['PS4', 'Wii', 'PS4', 'PC'],
        'Genre': ['Action', 'Sports', 'Action', 'Strategy'],
        'Publisher': ['Sony', 'Nintendo', 'Sony', 'Ubisoft'],
        'Critic_Score': [85, np.nan, 90, 70],
        'User_Score': ['8.5', 'tbd', '9.0', '7.5'],
        'Critic_Count': [50, 20, 60, 10],
        'User_Count': [200, np.nan, 300, 50],
        'Global_Sales': [2.0, 1.5, 3.0, 0.5]
    })

    pipeline = DataPipeline(top_n_publishers=5)
    processed_train = pipeline.fit_transform(train_data)

    assert not processed_train.isnull().values.any(), "Vẫn còn missing values trong output!"
    assert 'User_Score' in processed_train.columns, "Không tìm thấy cột User_Score!"
    assert 'Platform_PS4' in processed_train.columns, "Không có one-hot encoding!"
    print("test_datapipeline_basic đã chạy thành công")


def test_leakage_prevention():
    print("Running test_leakage_prevention...")
    train_df = pd.DataFrame({
        'Platform': ['PS4', 'PC'],
        'Genre': ['Action', 'Sports'],
        'Publisher': ['A', 'B'],
        'Critic_Score': [80, 60],
        'User_Score': ['8.0', '6.0'],
        'Critic_Count': [10, 20],
        'User_Count': [100, 200]
    })

    test_df = pd.DataFrame({
        'Platform': ['PS4', 'X360'],
        'Genre': ['Action', 'Action'],
        'Publisher': ['A', 'C'],
        'Critic_Score': [np.nan, 70],
        'User_Score': ['tbd', '7.0'],
        'Critic_Count': [15, 25],
        'User_Count': [150, 250]
    })

    pipeline = DataPipeline(top_n_publishers=2)
    train_out = pipeline.fit(train_df).transform(train_df)
    test_out = pipeline.transform(test_df)

    assert train_out.shape[1] == test_out.shape[1], "Shape mismatch! Test pipeline error."
    assert list(train_out.columns) == list(test_out.columns), "Column alignment error!"
    print("test_leakage_prevention đã chạy thành công")


def test_missing_expected_columns():
    print("Running test_missing_expected_columns...")
    train_df = pd.DataFrame({
        'Platform': ['PS4', 'PC'],
        'Genre': ['Action', 'Sports'],
        'Publisher': ['A', 'B'],
        'Critic_Score': [80, 60],
    })

    pipeline = DataPipeline(top_n_publishers=2)
    out = pipeline.fit_transform(train_df)

    assert set(['Critic_Count', 'User_Score', 'User_Count']).issubset(out.columns)
    assert not out.isna().any().any(), "Pipeline phải xử lý được cột thiếu trong input."
    print("test_missing_expected_columns đã chạy thành công")


if __name__ == "__main__":
    print("=== Bắt đầu kiểm thử DataPipeline ===")
    test_datapipeline_basic()
    test_leakage_prevention()
    test_missing_expected_columns()
    print("=== Tất cả unit tests đều thành công! ===")
