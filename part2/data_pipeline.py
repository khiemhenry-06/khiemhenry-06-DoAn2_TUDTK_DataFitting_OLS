import pandas as pd
import numpy as np
import warnings

class DataPipeline:
    def __init__(self, top_n_publishers=30):
        self.numerical_cols = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
        self.categorical_cols = ['Platform', 'Genre', 'Publisher']
        self.top_n_publishers = top_n_publishers
        
        # Dictionaris để lưu các thống kê học từ tập Train
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.top_publishers = []
        self.feature_columns_ = None 
        
    def _preprocess_basic(self, df):
        data = df.copy()
        if 'User_Score' in data.columns:
            data['User_Score'] = data['User_Score'].replace('tbd', np.nan)
            data['User_Score'] = pd.to_numeric(data['User_Score'], errors='coerce')
            
        for col in self.numerical_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        return data

    def fit(self, X, y=None):
        data = self._preprocess_basic(X)
        
        # Tính Median để điền khuyết sau này
        for col in self.numerical_cols:
            if col in data.columns:
                self.medians[col] = data[col].median()
                # Dự phòng trường hợp cột toàn NaN
                if pd.isna(self.medians[col]):
                    self.medians[col] = 0.0

        # Xử lý biến Publisher: Tìm top N publisher phổ biến nhất
        if 'Publisher' in data.columns:
            self.top_publishers = data['Publisher'].value_counts().nlargest(self.top_n_publishers).index.tolist()

        # Tạm thời điền khuyết để tính Mean/Std cho chuẩn hóa
        temp_data = data.copy()
        for col in self.numerical_cols:
            temp_data[col] = temp_data[col].fillna(self.medians[col])
            
        # Tính Mean và Std Dev cho Z-score standardization
        for col in self.numerical_cols:
            self.means[col] = temp_data[col].mean()
            self.stds[col] = temp_data[col].std()
            if self.stds[col] == 0 or pd.isna(self.stds[col]):
                self.stds[col] = 1.0
                
        # Dummy run to learn the encoded columns list
        dummy_out = self._transform_internal(X)
        self.feature_columns_ = dummy_out.columns.tolist()
        
        return self

    def _transform_internal(self, X):
        data = self._preprocess_basic(X)
        
        # Xử lý Điền khuyết Numerical (Imputation)
        for col in self.numerical_cols:
            fill_value = self.medians.get(col, 0.0)
            data[col] = data[col].fillna(fill_value)
            
        # Xử lý Rút gọn Publisher
        if 'Publisher' in data.columns and len(self.top_publishers) > 0:
            data['Publisher'] = data['Publisher'].apply(lambda x: x if x in self.top_publishers else 'Other')
            
        # Điền khuyết biến phân loại bằng 'Unknown'
        for col in self.categorical_cols:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        # Chuẩn hóa Numerical
        for col in self.numerical_cols:
            mean = self.means.get(col, 0.0)
            std = self.stds.get(col, 1.0)
            data[col] = (data[col] - mean) / std
            
        # One-Hot Encoding bằng pandas get_dummies
        df_encoded = pd.get_dummies(data, columns=self.categorical_cols, prefix=self.categorical_cols)
        
        cols_to_keep = self.numerical_cols.copy()
        dummy_cols = [c for c in df_encoded.columns if any(c.startswith(cat + "_") for cat in self.categorical_cols)]
        cols_to_keep.extend(dummy_cols)
        
        return df_encoded[cols_to_keep]

    def transform(self, X):
        if self.feature_columns_ is None:
            raise RuntimeError("Phải gọi fit() trước khi transform().")
            
        transformed = self._transform_internal(X)
        
        # Đảm bảo căn chỉnh 100% khớp với các cột lúc fit
        missing_cols = set(self.feature_columns_) - set(transformed.columns)
        for col in missing_cols:
            transformed[col] = 0
            
        final_df = transformed[self.feature_columns_]
        
        return final_df

    def fit_transform(self, X, y=None):
        """Gộp cả fit và transform trong 1 lần chạy"""
        return self.fit(X).transform(X)

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
    
    # Verify các điểm chính
    assert not processed_train.isnull().values.any(), "Vẫn còn missing values trong output!"
    assert 'User_Score' in processed_train.columns, "Không tìm thấy cột User_Score!"
    assert 'Platform_PS4' in processed_train.columns, "Không có one-hot encoding!"
    print(" test_datapipeline_basic đã chạy thành công")

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

if __name__ == "__main__":
    print("=== Bắt đầu kiểm thử DataPipeline ===")
    test_datapipeline_basic()
    test_leakage_prevention()
    print("=== Tất cả các Unit tests đều thành công! ===")
