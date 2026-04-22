import os
import sys
import torch
import numpy as np
from models.EnhancedTimeLLM import EnhancedTimeLLM
from data_provider.data_processor_enhanced import DataLoaderFactory
from experiments.experiment_runner import ExperimentConfig, ExperimentRunner

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_llm_external_model():
    """
    测试LLM外部因素模型
    """
    print("\n=== Testing LLM External Factor Model ===")
    
    # 创建测试配置
    class TestConfig:
        llm_model = 'BERT'
        llm_layers = 6
        llm_dim = 768
        d_model = 512
    
    config = TestConfig()
    
    # 导入LLM外部因素模型
    from models.llm_external_model import LLMExternalFactorModel
    
    # 创建模型
    llm_external_model = LLMExternalFactorModel(config)
    
    # 创建测试数据
    test_factors = [
        {
            'weather': {'temperature': 25.5, 'humidity': 65.0, 'wind_speed': 3.2},
            'holiday': {'is_holiday': True, 'holiday_name': 'National Day'},
            'time_features': {'hour': 10, 'dayofweek': 5, 'is_weekend': True}
        },
        {
            'weather': {'temperature': 18.3, 'humidity': 45.0, 'wind_speed': 2.1},
            'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
            'time_features': {'hour': 14, 'dayofweek': 2, 'is_weekend': False}
        }
    ]
    
    # 测试前向传播
    semantic_features, impact_scores = llm_external_model(test_factors)
    
    print(f"Semantic features shape: {semantic_features.shape}")
    print(f"Impact scores shape: {impact_scores.shape}")
    print(f"Impact scores: {impact_scores.detach().numpy().flatten()}")
    
    print("LLM External Factor Model test completed successfully!")

def test_fusion_modules():
    """
    测试融合模块
    """
    print("\n=== Testing Fusion Modules ===")
    
    # 创建测试数据
    B = 4
    L = 24
    D = 512
    
    temporal_features = torch.randn(B, L, D)
    semantic_features = torch.randn(B, D)
    impact_scores = torch.randn(B, 1)
    
    # 导入融合模块
    from models.fusion_modules import DynamicFusionModule
    
    # 创建动态融合模块
    dynamic_fusion = DynamicFusionModule(D, n_heads=8)
    
    # 测试前向传播
    fused, attention = dynamic_fusion(temporal_features, semantic_features, impact_scores)
    
    print(f"Dynamic Fusion Output shape: {fused.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    print("Fusion Modules test completed successfully!")

def test_data_processor():
    """
    测试数据处理器
    """
    print("\n=== Testing Data Processor ===")
    
    # 创建测试配置
    class TestConfig:
        seq_len = 96
        label_len = 48
        pred_len = 96
    
    config = TestConfig()
    
    # 创建数据处理器
    from data_provider.data_processor_enhanced import DataProcessor
    data_processor = DataProcessor(config)
    
    # 测试数据加载和预处理
    try:
        # 尝试加载数据
        df = data_processor.load_data('./dataset/数据/', 'processed_data.csv')
        print("Data loaded successfully!")
        
        # 预处理数据
        df = data_processor.preprocess_data(df)
        print("Data preprocessing completed successfully!")
        
        # 测试外部因素提取
        external_factors = data_processor.extract_external_factors(df, [0, 1, 2])
        print(f"External factors extracted: {len(external_factors)}")
        print(f"First external factor: {external_factors[0]}")
        
        print("Data Processor test completed successfully!")
        return True
    except Exception as e:
        print(f"Error in Data Processor test: {e}")
        print("Data Processor test completed with errors. Please check your data path.")
        return False

def test_enhanced_model():
    """
    测试增强版模型
    """
    print("\n=== Testing Enhanced TimeLLM Model ===")
    
    # 创建测试配置
    class TestConfig:
        task_name = 'long_term_forecast'
        pred_len = 96
        seq_len = 96
        label_len = 48
        d_model = 512
        d_ff = 2048
        n_heads = 8
        dropout = 0.1
        enc_in = 1
        llm_model = 'BERT'
        llm_layers = 6
        llm_dim = 768
        patch_len = 16
        stride = 8
        use_external_factors = True
        use_llm_semantic = True
        use_fusion = True
    
    config = TestConfig()
    
    # 创建模型
    model = EnhancedTimeLLM(config)
    
    # 创建测试输入
    B = 2
    T = config.seq_len
    N = config.enc_in
    
    x_enc = torch.randn(B, T, N)
    x_mark_enc = torch.randn(B, T, 5)  # 5 time features
    x_dec = torch.randn(B, config.label_len + config.pred_len, N)
    x_mark_dec = torch.randn(B, config.label_len + config.pred_len, 5)
    
    # 创建测试外部因素
    external_factors = [
        {
            'weather': {'temperature': 25.5, 'humidity': 65.0, 'wind_speed': 3.2},
            'holiday': {'is_holiday': True, 'holiday_name': 'National Day'},
            'time_features': {'hour': 10, 'dayofweek': 5, 'is_weekend': True}
        },
        {
            'weather': {'temperature': 18.3, 'humidity': 45.0, 'wind_speed': 2.1},
            'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
            'time_features': {'hour': 14, 'dayofweek': 2, 'is_weekend': False}
        }
    ]
    
    # 测试前向传播
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors)
    
    print(f"Model output shape: {output.shape}")
    print(f"Expected shape: ({B}, {config.pred_len}, {N})")
    
    print("Enhanced TimeLLM Model test completed successfully!")

def run_full_test():
    """
    运行完整测试
    """
    print("=====================================")
    print("Running Full Framework Test")
    print("=====================================")
    
    # 测试LLM外部因素模型
    test_llm_external_model()
    
    # 测试融合模块
    test_fusion_modules()
    
    # 测试数据处理器
    data_test_passed = test_data_processor()
    
    # 测试增强版模型
    test_enhanced_model()
    
    print("\n=====================================")
    print("Full Framework Test Completed")
    print("=====================================")
    
    if data_test_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests completed with errors. Please check your setup.")

def run_sample_experiment():
    """
    运行样本实验
    """
    print("\n=====================================")
    print("Running Sample Experiment")
    print("=====================================")
    
    # 创建配置
    config_dict = {
        'root_path': './dataset/数据/',
        'data_path': 'processed_data.csv',
        'save_path': './results/sample/',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 2,
        'early_stop': 5,
        'd_model': 512,
        'n_heads': 8,
        'llm_model': 'BERT',
        'llm_layers': 6,
        'llm_dim': 768,
        'patch_len': 16,
        'stride': 8,
        'dropout': 0.1,
        'enc_in': 1,
        'task_name': 'long_term_forecast'
    }
    
    # 创建保存目录
    os.makedirs(config_dict['save_path'], exist_ok=True)
    
    # 创建配置对象
    configs = ExperimentConfig(config_dict)
    
    # 保存配置
    configs.save(os.path.join(config_dict['save_path'], 'config.json'))
    
    # 创建实验运行器
    runner = ExperimentRunner(configs)
    
    # 运行单个实验
    try:
        metrics = runner.run_experiment('llm_fusion')
        print("Sample experiment completed successfully!")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"Error running sample experiment: {e}")
        print("Sample experiment completed with errors.")

if __name__ == '__main__':
    # 运行完整测试
    run_full_test()
    
    # 运行样本实验（可选）
    # run_sample_experiment()
