import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom_Enhanced(Dataset):
    """
    增强版自定义数据集：支持外部因素（天气、节假日）的加载和处理
    """
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', target='flow', scale=True, timeenc=0, freq='h', percent=100,
                 external_features=True):
        """
        初始化数据集
        
        参数:
            root_path: 数据根路径
            data_path: 数据文件路径
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M'多变量, 'S'单变量
            target: 目标变量名
            scale: 是否归一化
            timeenc: 时间编码方式
            freq: 频率
            percent: 数据百分比
            external_features: 是否使用外部特征
        """
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.target = target
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.external_features = external_features
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """读取并处理数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 数据集划分
        border1s = [0, int(len(df_raw)*0.7), int(len(df_raw)*0.8)]
        border2s = [int(len(df_raw)*0.7), int(len(df_raw)*0.8), len(df_raw)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        border1 = border1s[type_map[self.flag]]
        border2 = border2s[type_map[self.flag]]
        
        # 选择特征列
        if self.external_features:
            # 包含外部因素的特征
            cols_data = ['people', 'temperature', 'humidity', 'wind_speed', 'is_holiday']
        else:
            # 仅使用人流数据
            cols_data = ['people']
        
        # 确保列存在
        available_cols = [col for col in cols_data if col in df_raw.columns]
        if len(available_cols) < len(cols_data):
            missing_cols = set(cols_data) - set(available_cols)
            print(f"Warning: Missing columns {missing_cols}, using available columns only")
        
        df_data = df_raw[available_cols]
        
        # 归一化
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 提取时间特征
        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp['datetime'])
        
        if self.timeenc == 0:
            # 基本时间特征
            df_stamp['month'] = df_stamp['datetime'].dt.month
            df_stamp['day'] = df_stamp['datetime'].dt.day
            df_stamp['weekday'] = df_stamp['datetime'].dt.weekday
            df_stamp['hour'] = df_stamp['datetime'].dt.hour
            df_stamp['minute'] = df_stamp['datetime'].dt.minute
            data_stamp = df_stamp.drop(['datetime'], axis=1).values
        elif self.timeenc == 1:
            # 使用时间特征编码
            from utils.timefeatures import time_features
            data_stamp = time_features(df_stamp['datetime'].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # 保存原始数据用于提取外部因素
        self.raw_data = df_raw[border1:border2]

    def __getitem__(self, index):
        """获取单个样本"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反归一化"""
        return self.scaler.inverse_transform(data)
    
    def get_external_factors(self, index):
        """
        获取外部因素信息
        
        参数:
            index: 样本索引
        
        返回:
            dict: 外部因素字典
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 获取原始数据
        raw_slice = self.raw_data.iloc[s_begin:s_end]
        
        external_factors = {
            'weather': [],
            'holiday': [],
            'time_features': []
        }
        
        for idx in range(len(raw_slice)):
            row = raw_slice.iloc[idx]
            
            # 天气特征
            weather = {
                'temperature': row.get('temperature', 20.0),
                'humidity': row.get('humidity', 50.0),
                'wind_speed': row.get('wind_speed', 2.0)
            }
            external_factors['weather'].append(weather)
            
            # 节假日特征
            holiday = {
                'is_holiday': bool(row.get('is_holiday', 0)),
                'holiday_name': 'holiday' if row.get('is_holiday', 0) else 'ordinary day'
            }
            external_factors['holiday'].append(holiday)
            
            # 时间特征
            datetime_obj = pd.to_datetime(row['datetime'])
            time_feat = {
                'hour': datetime_obj.hour,
                'dayofweek': datetime_obj.weekday(),
                'is_weekend': datetime_obj.weekday() >= 5
            }
            external_factors['time_features'].append(time_feat)
        
        return external_factors


class Dataset_People_Flow_Enhanced(Dataset):
    """
    专门用于人流预测的增强版数据集
    """
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', target='people', scale=True, timeenc=0, freq='10min', percent=100):
        """
        初始化人流预测数据集
        
        参数:
            root_path: 数据根路径
            data_path: 数据文件路径
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M'多变量, 'S'单变量
            target: 目标变量名
            scale: 是否归一化
            timeenc: 时间编码方式
            freq: 频率（10min, h, d等）
            percent: 数据百分比
        """
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        """读取并处理人流数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 数据集划分
        border1s = [0, int(len(df_raw)*0.7), int(len(df_raw)*0.8)]
        border2s = [int(len(df_raw)*0.7), int(len(df_raw)*0.8), len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            # 多变量：人流 + 天气 + 节假日
            cols_data = ['people', 'temperature', 'humidity', 'wind_speed', 'is_holiday']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量：仅人流
            df_data = df_raw[[self.target]]
        
        # 归一化
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 提取时间特征
        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp['datetime'])
        
        if self.timeenc == 0:
            # 基本时间特征
            df_stamp['month'] = df_stamp['datetime'].dt.month
            df_stamp['day'] = df_stamp['datetime'].dt.day
            df_stamp['weekday'] = df_stamp['datetime'].dt.weekday
            df_stamp['hour'] = df_stamp['datetime'].dt.hour
            df_stamp['minute'] = df_stamp['datetime'].dt.minute
            data_stamp = df_stamp.drop(['datetime'], axis=1).values
        elif self.timeenc == 1:
            # 使用时间特征编码
            from utils.timefeatures import time_features
            data_stamp = time_features(df_stamp['datetime'].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # 保存原始数据用于提取外部因素
        self.raw_data = df_raw[border1:border2]

    def __getitem__(self, index):
        """获取单个样本"""
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        """反归一化"""
        return self.scaler.inverse_transform(data)
    
    def get_external_factors_batch(self, indices):
        """
        批量获取外部因素信息
        
        参数:
            indices: 样本索引列表
        
        返回:
            list: 外部因素字典列表
        """
        external_factors_batch = []
        
        for index in indices:
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            
            # 获取原始数据
            raw_slice = self.raw_data.iloc[s_begin:s_end]
            
            # 取中间时间点的外部因素
            mid_idx = len(raw_slice) // 2
            row = raw_slice.iloc[mid_idx]
            
            external_factors = {
                'weather': {
                    'temperature': row.get('temperature', 20.0),
                    'humidity': row.get('humidity', 50.0),
                    'wind_speed': row.get('wind_speed', 2.0)
                },
                'holiday': {
                    'is_holiday': bool(row.get('is_holiday', 0)),
                    'holiday_name': 'holiday' if row.get('is_holiday', 0) else 'ordinary day'
                },
                'time_features': {
                    'hour': pd.to_datetime(row['datetime']).hour,
                    'dayofweek': pd.to_datetime(row['datetime']).weekday,
                    'is_weekend': pd.to_datetime(row['datetime']).weekday >= 5
                }
            }
            
            external_factors_batch.append(external_factors)
        
        return external_factors_batch


def create_enhanced_dataset(root_path, data_path, flag='train', size=None, 
                           features='M', target='people', scale=True, timeenc=0, 
                           freq='10min', percent=100, dataset_type='people_flow'):
    """
    创建增强版数据集的工厂函数
    
    参数:
        root_path: 数据根路径
        data_path: 数据文件路径
        flag: 'train', 'val', 'test'
        size: [seq_len, label_len, pred_len]
        features: 'M'多变量, 'S'单变量
        target: 目标变量名
        scale: 是否归一化
        timeenc: 时间编码方式
        freq: 频率
        percent: 数据百分比
        dataset_type: 数据集类型 'people_flow' 或 'custom'
    
    返回:
        Dataset: 数据集实例
    """
    if dataset_type == 'people_flow':
        return Dataset_People_Flow_Enhanced(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=size,
            features=features,
            target=target,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
            percent=percent
        )
    else:
        return Dataset_Custom_Enhanced(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=size,
            features=features,
            target=target,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            external_features=True
        )
