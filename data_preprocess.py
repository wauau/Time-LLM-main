#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
用于执行完整的数据处理流程，包括时间特征提取、数据对齐和去噪等
"""

import argparse
from data_provider import complete_data_processing


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--input_people', type=str, default='./dataset/数据/people_10_12_new.csv',
                        help='人流量数据文件路径')
    parser.add_argument('--input_weather', type=str, default='./dataset/数据/weather_10_12_10min_utf8.csv',
                        help='天气数据文件路径')
    parser.add_argument('--input_events', type=str, default='./dataset/数据/events_10_12_utf8.csv',
                        help='节假日数据文件路径')
    parser.add_argument('--output', type=str, default='./dataset/数据/processed_data.csv',
                        help='处理后的数据保存路径')
    parser.add_argument('--freq', type=str, default='10min',
                        help='数据对齐的时间频率')
    parser.add_argument('--denoise_method', type=str, default='moving_average',
                        choices=['moving_average', 'exponential_smoothing', 'outlier_detection'],
                        help='去噪方法')
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    print("开始数据预处理...")
    print(f"人流量数据: {args.input_people}")
    print(f"天气数据: {args.input_weather}")
    print(f"节假日数据: {args.input_events}")
    print(f"输出路径: {args.output}")
    print(f"时间频率: {args.freq}")
    print(f"去噪方法: {args.denoise_method}")
    
    # 执行数据处理
    processed_data = complete_data_processing()
    
    print("\n数据预处理完成！")
    print(f"处理后的数据形状: {processed_data.shape}")
    print(f"处理后的数据列: {list(processed_data.columns)}")
    print(f"数据已保存到: {args.output}")


if __name__ == '__main__':
    main()
