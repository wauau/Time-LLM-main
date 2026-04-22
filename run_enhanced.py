import argparse
import os
import torch
import numpy as np
import random
import time

from exp.exp_main import Exp_Main
from models.EnhancedTimeLLM import create_enhanced_model
from data_provider.data_loader_enhanced import create_enhanced_dataset


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Exp_Enhanced(Exp_Main):
    """
    增强版实验类：继承自Exp_Main，使用增强版模型和数据集
    """
    def __init__(self, args):
        super(Exp_Enhanced, self).__init__(args)
        self.use_cross_attention = getattr(args, 'use_cross_attention', True)
        self.use_similarity_alignment = not self.use_cross_attention
    
    def _build_model(self):
        """构建增强版模型"""
        from models.EnhancedTimeLLM import EnhancedTimeLLM
        
        # 设置use_cross_attention属性
        self.args.use_cross_attention = self.use_cross_attention
        
        model = EnhancedTimeLLM(self.args)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag):
        """获取增强版数据集"""
        data_set, data_loader = None, None
        
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size
        
        # 创建增强版数据集
        data_set = create_enhanced_dataset(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=self.args.scale,
            timeenc=self.args.timeenc,
            freq=self.args.freq,
            percent=self.args.percent,
            dataset_type='people_flow'
        )
        
        print(flag, len(data_set))
        
        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        
        return data_set, data_loader


def main():
    parser = argparse.ArgumentParser(description='Enhanced TimeLLM Training')
    
    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='people_flow_96_96', help='model id')
    parser.add_argument('--model', type=str, default='EnhancedTimeLLM', help='model name')
    
    # 数据配置
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/数据/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='processed_data.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='people', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='10min', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # 模型配置
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=5, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # LLM配置
    parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model type')
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM embedding dimension')
    parser.add_argument('--llm_layers', type=int, default=6, help='number of LLM layers to use')
    parser.add_argument('--prompt_domain', type=int, default=1, help='whether to use domain-specific prompt')
    parser.add_argument('--content', type=str, default='People flow forecasting with external factors including weather and holidays', help='task description')
    
    # 增强功能配置
    parser.add_argument('--use_cross_attention', action='store_true', help='use cross-attention fusion')
    parser.add_argument('--use_similarity_alignment', action='store_true', help='use similarity alignment fusion')
    
    # 训练配置
    parser.add_argument('--num_workers', type=int, default=10, help='data loader workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')
    
    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    # 其他配置
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--percent', type=int, default=100, help='percentage of training data')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验对象
    exp = Exp_Enhanced(args)
    
    if args.is_training:
        print('>>>>>>>开始训练>>>>>>>')
        train_losses = exp.train(setting)
        
        print('>>>>>>>开始测试>>>>>>>')
        mse, mae = exp.test(setting)
        print(f'Test MSE: {mse:.4f}, Test MAE: {mae:.4f}')
    else:
        print('>>>>>>>开始预测>>>>>>>')
        prediction = exp.predict(setting, True)
        print(f'Prediction shape: {prediction.shape}')


if __name__ == '__main__':
    main()
