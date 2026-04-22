import torch
from models.model_full import FullModel
from models.TimeLLM import Model as TimeLLM
from transformers import BertModel, BertTokenizerFast, BertConfig

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
        self.llm_layers = 2
        self.prompt_domain = False
        self.content = 'People flow forecasting'

# 初始化LLM模型和tokenizer
config = TestConfig()

# 创建BERT配置
bert_config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=2,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    output_attentions=True,
    output_hidden_states=True
)

# 创建BERT模型
llm_model = BertModel(bert_config)

# 创建本地tokenizer，不依赖下载
tokenizer = BertTokenizerFast(
    vocab_file=None,
    do_lower_case=True,
    strip_accents=True,
    tokenize_chinese_chars=False,
    wordpiece_prefix="##"
)
# 添加特殊 tokens
special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
tokenizer.add_special_tokens(special_tokens)

# 创建完整模型
full_model = FullModel(config, llm_model, tokenizer)

# 创建测试数据
batch_size = 2
seq_len = config.seq_len
pred_len = config.pred_len
enc_in = config.enc_in

x = torch.randn(batch_size, seq_len, enc_in)

# 创建外因输入数据
external_data = [
    {
        "date": "2024-05-01",
        "weather": "rainy",
        "holiday": "Labor Day"
    },
    {
        "date": "2024-05-02",
        "weather": "sunny",
        "holiday": "None"
    }
]

print("Testing FullModel...")

# 测试前向传播
try:
    output, alpha = full_model(x, external_data)
    print(f"Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {pred_len}, {enc_in})")
    print(f"Alpha shape: {alpha.shape}")
    print(f"Alpha values:")
    print(alpha)
    print("\nAll tests passed successfully!")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
