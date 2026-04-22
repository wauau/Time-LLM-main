import pandas as pd
import numpy as np
from scipy import signal


def extract_time_features(df, time_col='timestamp'):
    """
    提取时间特征
    
    参数:
        df (pd.DataFrame): 输入数据框
        time_col (str): 时间列名称
    
    返回:
        pd.DataFrame: 包含时间特征的数据框
    """
    # 确保时间列为datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 基本时间特征
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['dayofweek'] = df[time_col].dt.dayofweek  # 0-6，0表示周一
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 季节特征
    def get_season(month):
        if month in [3,4,5]:
            return 1  # 春季
        elif month in [6,7,8]:
            return 2  # 夏季
        elif month in [9,10,11]:
            return 3  # 秋季
        else:
            return 4  # 冬季
    df['season'] = df['month'].apply(get_season)
    
    # 高峰期特征（假设8-9点和17-18点为高峰期）
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in [8,9,17,18] else 0)
    
    return df


def extract_holiday_features(df, time_col='timestamp', holiday_file='./dataset/数据/events_10_12_utf8.csv'):
    """
    提取节假日特征
    
    参数:
        df (pd.DataFrame): 输入数据框
        time_col (str): 时间列名称
        holiday_file (str): 节假日数据文件路径
    
    返回:
        pd.DataFrame: 包含节假日特征的数据框
    """
    # 读取节假日数据
    holidays = pd.read_csv(holiday_file)
    holidays['date'] = pd.to_datetime(holidays['date'])
    
    # 合并节假日信息
    df['date_only'] = df[time_col].dt.date
    holidays['date_only'] = holidays['date'].dt.date
    
    df = df.merge(holidays[['date_only', 'event_name']], on='date_only', how='left')
    df['is_holiday'] = df['event_name'].notna().astype(int)
    df['holiday_type'] = df['event_name'].fillna('普通日')
    
    # 清理临时列
    df.drop(['date_only', 'event_name'], axis=1, inplace=True)
    
    return df


def align_data(people_data, weather_data, time_col='timestamp', freq='10min'):
    """
    对齐人流量和天气数据
    
    参数:
        people_data (pd.DataFrame): 人流量数据
        weather_data (pd.DataFrame): 天气数据
        time_col (str): 时间列名称
        freq (str): 重采样频率
    
    返回:
        pd.DataFrame: 对齐后的数据
    """
    # 确保时间列为datetime类型，并统一时区
    people_data[time_col] = pd.to_datetime(people_data[time_col])
    weather_data[time_col] = pd.to_datetime(weather_data[time_col]).dt.tz_localize(None)  # 移除时区信息
    
    # 分离数值列和非数值列
    people_numeric_cols = people_data.select_dtypes(include=['number']).columns.tolist()
    people_numeric_cols.append(time_col)
    
    weather_numeric_cols = weather_data.select_dtypes(include=['number']).columns.tolist()
    weather_numeric_cols.append(time_col)
    
    # 按指定频率重采样（只对数值列）
    people_resampled = people_data[people_numeric_cols].set_index(time_col).resample(freq).mean().reset_index()
    weather_resampled = weather_data[weather_numeric_cols].set_index(time_col).resample(freq).mean().reset_index()
    
    # 合并数据
    aligned_data = pd.merge(people_resampled, weather_resampled, on=time_col, how='outer')
    
    # 处理缺失值
    aligned_data = aligned_data.sort_values(time_col).ffill().bfill()
    
    return aligned_data


def denoise_data(df, value_col='people_count', method='moving_average'):
    """
    数据去噪
    
    参数:
        df (pd.DataFrame): 输入数据框
        value_col (str): 需要去噪的列名
        method (str): 去噪方法，可选值: 'moving_average', 'exponential_smoothing', 'outlier_detection'
    
    返回:
        pd.DataFrame: 去噪后的数据
    """
    if method == 'moving_average':
        # 移动平均法
        df[f'{value_col}_denoised'] = df[value_col].rolling(window=3, min_periods=1).mean()
    elif method == 'exponential_smoothing':
        # 指数平滑法
        df[f'{value_col}_denoised'] = df[value_col].ewm(alpha=0.3).mean()
    elif method == 'outlier_detection':
        # 异常值检测与处理
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 替换异常值为上下界
        df[f'{value_col}_denoised'] = df[value_col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def complete_data_processing():
    """
    完整的数据处理流程
    
    返回:
        pd.DataFrame: 处理后的数据
    """
    # 1. 读取数据
    people_data = pd.read_csv('./dataset/数据/people_10_12_new.csv')
    weather_data = pd.read_csv('./dataset/数据/weather_10_12_10min_utf8.csv')
    
    # 2. 提取时间特征
    people_data = extract_time_features(people_data, 'datetime')
    weather_data = extract_time_features(weather_data, 'timestamp')
    
    # 3. 提取节假日特征
    people_data = extract_holiday_features(people_data, 'datetime')
    
    # 4. 数据对齐
    # 先将天气数据的时间列重命名为datetime，以便对齐
    weather_data.rename(columns={'timestamp': 'datetime'}, inplace=True)
    aligned_data = align_data(people_data, weather_data, 'datetime', '10min')
    
    # 5. 数据去噪
    aligned_data = denoise_data(aligned_data, 'people', 'moving_average')
    
    # 6. 保存处理后的数据
    aligned_data.to_csv('./dataset/数据/processed_data.csv', index=False)
    
    return aligned_data


if __name__ == '__main__':
    print("开始数据处理...")
    processed_data = complete_data_processing()
    print("数据处理完成！")
    print(f"处理后的数据形状: {processed_data.shape}")
    print(f"处理后的数据列: {list(processed_data.columns)}")
