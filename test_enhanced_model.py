import torch
from models.EnhancedTimeLLM import create_enhanced_model

# 创建测试配置
class TestConfig:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.pred_len = 24
        self.seq_len = 96
        self.d_ff = 2048
        self.llm_dim = 768
        self.patch_len = 16
        self.stride = 8
        self.d_model = 512
        self.enc_in = 1
        self.dropout = 0.1
        self.n_heads = 8
        self.llm_model = 'BERT'
        self.llm_layers = 6
        # 测试不同的实验模式
        self.experiment_mode = 'full'  # full, no_external, simple_concat, llm_only
        self.fusion_mode = 'attention'  # attention, gate, concat

# 创建模型
config = TestConfig()
model = create_enhanced_model(config)

# 创建测试数据
batch_size = 4
seq_len = config.seq_len
pred_len = config.pred_len
enc_in = config.enc_in

x_enc = torch.randn(batch_size, seq_len, enc_in)
x_mark_enc = torch.randn(batch_size, seq_len, 4)  # 时间特征 + 外部因素
x_dec = torch.randn(batch_size, pred_len, enc_in)
x_mark_dec = torch.randn(batch_size, pred_len, 4)

# 创建外部因素数据
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
    },
    {
        'weather': {'temperature': 22.0, 'humidity': 55.0, 'wind_speed': 2.5},
        'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
        'time_features': {'hour': 8, 'dayofweek': 1, 'is_weekend': False}
    },
    {
        'weather': {'temperature': 28.0, 'humidity': 70.0, 'wind_speed': 1.8},
        'holiday': {'is_holiday': True, 'holiday_name': 'Weekend'},
        'time_features': {'hour': 16, 'dayofweek': 6, 'is_weekend': True}
    }
]

print("Testing EnhancedTimeLLM model...")
print(f"Model configuration: experiment_mode={config.experiment_mode}, fusion_mode={config.fusion_mode}")

# 测试前向传播
try:
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors)
    print(f"Forward pass successful!")
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {pred_len}, {enc_in})")
    
    # 测试不同的实验模式
    print("\nTesting different experiment modes:")
    
    # 测试无外因模式
    config.experiment_mode = 'no_external'
    model_no_external = create_enhanced_model(config)
    output_no_external = model_no_external(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"No external factors mode: Output shape = {output_no_external.shape}")
    
    # 测试简单拼接模式
    config.experiment_mode = 'simple_concat'
    model_simple_concat = create_enhanced_model(config)
    output_simple_concat = model_simple_concat(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Simple concat mode: Output shape = {output_simple_concat.shape}")
    
    # 测试LLM外因模式
    config.experiment_mode = 'llm_only'
    model_llm_only = create_enhanced_model(config)
    output_llm_only = model_llm_only(x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors)
    print(f"LLM only mode: Output shape = {output_llm_only.shape}")
    
    print("\nAll tests passed successfully!")

except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
