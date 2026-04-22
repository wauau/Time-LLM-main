import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    统一数据处理器：负责数据加载、预处理和特征构建
    """
    def __init__(self, configs):
        self.configs = configs
        self.scaler = None
        self.external_scaler = None
    
    def load_data(self, root_path, data_path):
        """
        加载数据
        
        参数:
            root_path: 数据根路径
            data_path: 数据文件路径
        
        返回:
            df: 加载的数据DataFrame
        """
        file_path = os.path.join(root_path, data_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Data columns: {list(df.columns)}")
        
        return df
    
    def preprocess_data(self, df):
        """
        预处理数据
        
        参数:
            df: 原始数据DataFrame
        
        返回:
            df: 预处理后的数据
        """
        # 确保时间列存在
        if 'datetime' not in df.columns:
            raise ValueError("Dataframe must contain 'datetime' column")
        
        # 转换时间列
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 排序
        df = df.sort_values('datetime')
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 提取时间特征
        df = self._extract_time_features(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """
        处理缺失值
        """
        # 数值列填充
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        # 非数值列填充
        non_numeric_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
        for col in non_numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _extract_time_features(self, df):
        """
        提取时间特征
        """
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        
        return df
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.1):
        """
        划分数据集
        
        参数:
            df: 预处理后的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        返回:
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
        """
        total_len = len(df)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_df = df.iloc[:train_len]
        val_df = df.iloc[train_len:train_len+val_len]
        test_df = df.iloc[train_len+val_len:]
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def normalize_data(self, train_df, val_df, test_df, target_col='people'):
        """
        归一化数据
        
        参数:
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
            target_col: 目标列
        
        返回:
            train_data: 归一化后的训练数据
            val_data: 归一化后的验证数据
            test_data: 归一化后的测试数据
        """
        # 选择特征列
        feature_cols = [target_col, 'temperature', 'humidity', 'wind_speed', 'is_holiday']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        print(f"Using features: {available_cols}")
        
        # 初始化scaler
        self.scaler = StandardScaler()
        
        # 拟合训练数据
        train_features = train_df[available_cols].values
        self.scaler.fit(train_features)
        
        # 转换数据
        train_data = self.scaler.transform(train_features)
        val_data = self.scaler.transform(val_df[available_cols].values)
        test_data = self.scaler.transform(test_df[available_cols].values)
        
        return train_data, val_data, test_data
    
    def extract_external_factors(self, df, indices):
        """
        提取外部因素
        
        参数:
            df: 数据DataFrame
            indices: 索引列表
        
        返回:
            external_factors: 外部因素列表
        """
        external_factors = []
        
        for idx in indices:
            if idx < len(df):
                row = df.iloc[idx]
                factors = {
                    'weather': {
                        'temperature': row.get('temperature', 20.0),
                        'humidity': row.get('humidity', 50.0),
                        'wind_speed': row.get('wind_speed', 2.0)
                    },
                    'holiday': {
                        'is_holiday': bool(row.get('is_holiday', 0)),
                        'holiday_name': row.get('holiday_name', 'ordinary day') if row.get('is_holiday', 0) else 'ordinary day'
                    },
                    'time_features': {
                        'hour': row.get('hour', 12),
                        'dayofweek': row.get('dayofweek', 0),
                        'is_weekend': bool(row.get('is_weekend', 0))
                    }
                }
            else:
                # 默认值
                factors = {
                    'weather': {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 2.0},
                    'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
                    'time_features': {'hour': 12, 'dayofweek': 0, 'is_weekend': False}
                }
            
            external_factors.append(factors)
        
        return external_factors


class EnhancedDataset(Dataset):
    """
    增强版数据集：支持外部因素和LLM语义特征
    """
    def __init__(self, df, data, seq_len, label_len, pred_len, external_features=True):
        """
        初始化数据集
        
        参数:
            df: 原始数据DataFrame
            data: 归一化后的数据
            seq_len: 输入序列长度
            label_len: 标签序列长度
            pred_len: 预测序列长度
            external_features: 是否使用外部特征
        """
        self.df = df
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.external_features = external_features
        self.tot_len = len(data) - seq_len - pred_len + 1
    
    def __len__(self):
        return self.tot_len
    
    def __getitem__(self, index):
        """
        获取单个样本
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 提取序列数据
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end, [0]]  # 只取目标列
        
        # 提取时间特征
        seq_x_mark = self._extract_time_mark(s_begin, s_end)
        seq_y_mark = self._extract_time_mark(r_begin, r_end)
        
        # 提取外部因素（用于LLM处理）
        external_factors = None
        if self.external_features:
            # 取序列中间点的外部因素
            mid_idx = s_begin + self.seq_len // 2
            external_factors = self._extract_external_factors(mid_idx)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, external_factors
    
    def _extract_time_mark(self, start, end):
        """
        提取时间标记
        """
        if start >= len(self.df) or end > len(self.df):
            # 返回默认值
            return np.zeros((end - start, 5))  # 5个时间特征
        
        time_features = self.df.iloc[start:end][['month', 'day', 'dayofweek', 'hour', 'is_weekend']].values
        return time_features
    
    def _extract_external_factors(self, idx):
        """
        提取外部因素
        """
        if idx >= len(self.df):
            # 返回默认值
            return {
                'weather': {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 2.0},
                'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
                'time_features': {'hour': 12, 'dayofweek': 0, 'is_weekend': False}
            }
        
        row = self.df.iloc[idx]
        return {
            'weather': {
                'temperature': row.get('temperature', 20.0),
                'humidity': row.get('humidity', 50.0),
                'wind_speed': row.get('wind_speed', 2.0)
            },
            'holiday': {
                'is_holiday': bool(row.get('is_holiday', 0)),
                'holiday_name': row.get('holiday_name', 'ordinary day') if row.get('is_holiday', 0) else 'ordinary day'
            },
            'time_features': {
                'hour': row.get('hour', 12),
                'dayofweek': row.get('dayofweek', 0),
                'is_weekend': bool(row.get('is_weekend', 0))
            }
        }
    
    def get_external_factors_batch(self, indices):
        """
        批量获取外部因素
        """
        external_factors = []
        for idx in indices:
            if idx < self.tot_len:
                mid_idx = idx + self.seq_len // 2
                factors = self._extract_external_factors(mid_idx)
            else:
                # 默认值
                factors = {
                    'weather': {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 2.0},
                    'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
                    'time_features': {'hour': 12, 'dayofweek': 0, 'is_weekend': False}
                }
            external_factors.append(factors)
        return external_factors


class FeatureBuilder:
    """
    特征构建器：负责构建各种特征
    """
    def __init__(self, configs):
        self.configs = configs
    
    def build_temporal_features(self, data):
        """
        构建时序特征
        
        参数:
            data: 输入数据
        
        返回:
            temporal_features: 时序特征
        """
        # 这里可以添加更复杂的时序特征提取
        return data
    
    def build_external_features(self, external_factors):
        """
        构建外部因素特征
        
        参数:
            external_factors: 外部因素
        
        返回:
            external_features: 外部因素特征
        """
        # 这里可以添加外部因素特征的构建
        return external_factors
    
    def build_combined_features(self, temporal_features, semantic_features, impact_scores):
        """
        构建组合特征
        
        参数:
            temporal_features: 时序特征
            semantic_features: 语义特征
            impact_scores: 影响程度分数
        
        返回:
            combined_features: 组合特征
        """
        # 这里可以添加特征组合逻辑
        return temporal_features


class DataLoaderFactory:
    """
    数据加载器工厂：创建各种数据加载器
    """
    def __init__(self, configs):
        self.configs = configs
        self.data_processor = DataProcessor(configs)
    
    def create_dataloaders(self, root_path, data_path, batch_size=32):
        """
        创建数据加载器
        
        参数:
            root_path: 数据根路径
            data_path: 数据文件路径
            batch_size: 批次大小
        
        返回:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            scaler: 归一化器
        """
        # 加载数据
        df = self.data_processor.load_data(root_path, data_path)
        
        # 预处理数据
        df = self.data_processor.preprocess_data(df)
        
        # 划分数据集
        train_df, val_df, test_df = self.data_processor.split_data(df)
        
        # 归一化数据
        train_data, val_data, test_data = self.data_processor.normalize_data(
            train_df, val_df, test_df, target_col='people'
        )
        
        # 创建数据集
        train_dataset = EnhancedDataset(
            train_df, train_data, 
            self.configs.seq_len, 
            self.configs.label_len, 
            self.configs.pred_len
        )
        
        val_dataset = EnhancedDataset(
            val_df, val_data, 
            self.configs.seq_len, 
            self.configs.label_len, 
            self.configs.pred_len
        )
        
        test_dataset = EnhancedDataset(
            test_df, test_data, 
            self.configs.seq_len, 
            self.configs.label_len, 
            self.configs.pred_len
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader, self.data_processor.scaler


# 测试代码
if __name__ == '__main__':
    # 创建测试配置
    class TestConfig:
        seq_len = 96
        label_len = 48
        pred_len = 96
    
    config = TestConfig()
    
    # 创建数据处理器
    data_processor = DataProcessor(config)
    
    # 测试数据加载和预处理
    try:
        df = data_processor.load_data('./dataset/数据/', 'processed_data.csv')
        df = data_processor.preprocess_data(df)
        print("Data preprocessing completed successfully!")
        
        # 测试数据集划分
        train_df, val_df, test_df = data_processor.split_data(df)
        
        # 测试归一化
        train_data, val_data, test_data = data_processor.normalize_data(train_df, val_df, test_df)
        print("Data normalization completed successfully!")
        
        # 测试外部因素提取
        external_factors = data_processor.extract_external_factors(df, [0, 1, 2])
        print(f"External factors extracted: {len(external_factors)}")
        print(f"First external factor: {external_factors[0]}")
        
        # 测试数据集创建
        dataset = EnhancedDataset(df, train_data, config.seq_len, config.label_len, config.pred_len)
        print(f"Dataset length: {len(dataset)}")
        
        # 测试数据加载
        seq_x, seq_y, seq_x_mark, seq_y_mark, external_factors = dataset[0]
        print(f"Sample shapes:")
        print(f"  seq_x: {seq_x.shape}")
        print(f"  seq_y: {seq_y.shape}")
        print(f"  seq_x_mark: {seq_x_mark.shape}")
        print(f"  seq_y_mark: {seq_y_mark.shape}")
        print(f"  External factors: {external_factors is not None}")
        
        print("\nData processing test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Test completed with errors. Please check your data path.")
