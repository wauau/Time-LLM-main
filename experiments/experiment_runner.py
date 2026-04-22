import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from models.llm_external_model import LLMSemanticFeatureGenerator
from models.fusion_modules import DynamicFusionModule
from data_provider.data_processor_enhanced import DataLoaderFactory


class ExperimentConfig:
    """
    实验配置类
    """
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
    
    def save(self, path):
        """
        保存配置
        """
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """
        加载配置
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


class ModelEvaluator:
    """
    模型评估器
    """
    def __init__(self):
        pass
    
    def evaluate(self, y_true, y_pred):
        """
        评估模型性能
        
        参数:
            y_true: 真实值
            y_pred: 预测值
        
        返回:
            metrics: 评估指标字典
        """
        # 确保数据形状正确
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 计算评估指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def print_metrics(self, metrics, experiment_name):
        """
        打印评估指标
        """
        print(f"\n=== {experiment_name} Evaluation Metrics ===")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R²: {metrics['R2']:.4f}")
        print("=====================================")
    
    def save_metrics(self, metrics, path):
        """
        保存评估指标
        """
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)


class ExperimentRunner:
    """
    实验运行器
    """
    def __init__(self, configs):
        self.configs = configs
        self.evaluator = ModelEvaluator()
        self.data_loader_factory = DataLoaderFactory(configs)
    
    def run_experiment(self, experiment_type):
        """
        运行实验
        
        参数:
            experiment_type: 实验类型
                - 'no_external': 无外因
                - 'concat_external': 仅拼接外因
                - 'llm_semantic': LLM语义外因
                - 'llm_fusion': LLM+融合机制
        
        返回:
            metrics: 评估指标
        """
        print(f"\n=====================================")
        print(f"Running experiment: {experiment_type}")
        print(f"=====================================")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader, scaler = self.data_loader_factory.create_dataloaders(
            root_path=self.configs.root_path,
            data_path=self.configs.data_path,
            batch_size=self.configs.batch_size
        )
        
        # 创建模型
        model = self._create_model(experiment_type)
        
        # 训练模型
        self._train_model(model, train_loader, val_loader)
        
        # 测试模型
        metrics = self._test_model(model, test_loader, scaler)
        
        # 打印和保存指标
        self.evaluator.print_metrics(metrics, experiment_type)
        
        # 保存模型
        model_save_path = os.path.join(self.configs.save_path, f"model_{experiment_type}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        # 保存指标
        metrics_save_path = os.path.join(self.configs.save_path, f"metrics_{experiment_type}.json")
        self.evaluator.save_metrics(metrics, metrics_save_path)
        print(f"Metrics saved to: {metrics_save_path}")
        
        return metrics
    
    def _create_model(self, experiment_type):
        """
        创建模型
        """
        # 这里需要根据experiment_type创建不同的模型
        # 暂时返回一个基础模型
        from models.EnhancedTimeLLM import EnhancedTimeLLM
        
        model = EnhancedTimeLLM(self.configs)
        
        # 根据实验类型配置模型
        if experiment_type == 'no_external':
            model.use_external_factors = False
            model.use_llm_semantic = False
            model.use_fusion = False
        elif experiment_type == 'concat_external':
            model.use_external_factors = True
            model.use_llm_semantic = False
            model.use_fusion = False
        elif experiment_type == 'llm_semantic':
            model.use_external_factors = True
            model.use_llm_semantic = True
            model.use_fusion = False
        elif experiment_type == 'llm_fusion':
            model.use_external_factors = True
            model.use_llm_semantic = True
            model.use_fusion = True
        
        return model
    
    def _train_model(self, model, train_loader, val_loader):
        """
        训练模型
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.configs.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        early_stop_count = 0
        
        for epoch in range(self.configs.epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                seq_x, seq_y, seq_x_mark, seq_y_mark, external_factors = batch
                
                # 移动到设备
                seq_x = seq_x.float().to(device)
                seq_y = seq_y.float().to(device)
                seq_x_mark = seq_x_mark.float().to(device)
                seq_y_mark = seq_y_mark.float().to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(seq_x, seq_x_mark, seq_y, seq_y_mark, external_factors)
                
                # 计算损失
                loss = criterion(outputs, seq_y[:, -self.configs.pred_len:, :])
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    seq_x, seq_y, seq_x_mark, seq_y_mark, external_factors = batch
                    
                    # 移动到设备
                    seq_x = seq_x.float().to(device)
                    seq_y = seq_y.float().to(device)
                    seq_x_mark = seq_x_mark.float().to(device)
                    seq_y_mark = seq_y_mark.float().to(device)
                    
                    # 前向传播
                    outputs = model(seq_x, seq_x_mark, seq_y, seq_y_mark, external_factors)
                    
                    # 计算损失
                    loss = criterion(outputs, seq_y[:, -self.configs.pred_len:, :])
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{self.configs.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(self.configs.save_path, "best_model.pt"))
            else:
                early_stop_count += 1
                if early_stop_count >= self.configs.early_stop:
                    print("Early stopping triggered!")
                    break
    
    def _test_model(self, model, test_loader, scaler):
        """
        测试模型
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                seq_x, seq_y, seq_x_mark, seq_y_mark, external_factors = batch
                
                # 移动到设备
                seq_x = seq_x.float().to(device)
                seq_y = seq_y.float().to(device)
                seq_x_mark = seq_x_mark.float().to(device)
                seq_y_mark = seq_y_mark.float().to(device)
                
                # 前向传播
                outputs = model(seq_x, seq_x_mark, seq_y, seq_y_mark, external_factors)
                
                # 反归一化
                outputs_np = outputs.cpu().numpy()
                seq_y_np = seq_y[:, -self.configs.pred_len:, :].cpu().numpy()
                
                # 反归一化（只反归一化目标列）
                # 假设第一列是目标列
                outputs_denorm = scaler.inverse_transform(
                    np.concatenate([outputs_np, np.zeros((outputs_np.shape[0], outputs_np.shape[1], scaler.n_features_in_ - 1))], axis=-1)
                )[:, :, 0]
                
                seq_y_denorm = scaler.inverse_transform(
                    np.concatenate([seq_y_np, np.zeros((seq_y_np.shape[0], seq_y_np.shape[1], scaler.n_features_in_ - 1))], axis=-1)
                )[:, :, 0]
                
                y_true.extend(seq_y_denorm.flatten())
                y_pred.extend(outputs_denorm.flatten())
        
        # 计算评估指标
        metrics = self.evaluator.evaluate(y_true, y_pred)
        
        return metrics
    
    def run_all_experiments(self):
        """
        运行所有实验
        """
        experiment_types = [
            'no_external',
            'concat_external',
            'llm_semantic',
            'llm_fusion'
        ]
        
        all_metrics = {}
        
        for experiment_type in experiment_types:
            metrics = self.run_experiment(experiment_type)
            all_metrics[experiment_type] = metrics
        
        # 保存所有指标
        metrics_save_path = os.path.join(self.configs.save_path, "all_metrics.json")
        with open(metrics_save_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # 打印对比结果
        self._print_experiment_comparison(all_metrics)
        
        return all_metrics
    
    def _print_experiment_comparison(self, all_metrics):
        """
        打印实验对比结果
        """
        print("\n=====================================")
        print("Experiment Comparison")
        print("=====================================")
        print(f"{'Experiment':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        
        for experiment, metrics in all_metrics.items():
            print(f"{experiment:<20} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['R2']:<10.4f}")
        
        print("-" * 50)
        
        # 找出最佳模型
        best_mae = float('inf')
        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_model_mae = ''
        best_model_rmse = ''
        best_model_r2 = ''
        
        for experiment, metrics in all_metrics.items():
            if metrics['MAE'] < best_mae:
                best_mae = metrics['MAE']
                best_model_mae = experiment
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_model_rmse = experiment
            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_model_r2 = experiment
        
        print(f"Best model (MAE): {best_model_mae} ({best_mae:.4f})")
        print(f"Best model (RMSE): {best_model_rmse} ({best_rmse:.4f})")
        print(f"Best model (R²): {best_model_r2} ({best_r2:.4f})")
        print("=====================================")


# 测试代码
if __name__ == '__main__':
    # 创建配置
    config_dict = {
        'root_path': './dataset/数据/',
        'data_path': 'processed_data.csv',
        'save_path': './results/',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stop': 10,
        'd_model': 512,
        'n_heads': 8,
        'llm_model': 'BERT',
        'llm_layers': 6,
        'llm_dim': 768
    }
    
    # 创建保存目录
    os.makedirs(config_dict['save_path'], exist_ok=True)
    
    # 创建配置对象
    configs = ExperimentConfig(config_dict)
    
    # 保存配置
    configs.save(os.path.join(config_dict['save_path'], 'config.json'))
    
    # 创建实验运行器
    runner = ExperimentRunner(configs)
    
    # 运行所有实验
    try:
        all_metrics = runner.run_all_experiments()
        print("\nAll experiments completed successfully!")
    except Exception as e:
        print(f"Error running experiments: {e}")
