"""
配置管理类
@author: llf
"""
import yaml
import os
from easydict import EasyDict


class Config:
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.cfg = self._load_config()
        self._setup_paths()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return EasyDict(cfg)

    def _setup_paths(self):
        """设置实验相关路径"""
        exp_name = self.cfg.experiments.exp_name
        self.exp_dir = f'experiments/{exp_name}'
        self.checkpoint_path = f'{self.exp_dir}/checkpoints'
        self.log_path = f'{self.exp_dir}/logs'

        # 创建必要的目录
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def update_exp(self, exp_name):
        """更新实验名称"""
        self.cfg.experiments.exp_name = exp_name
        self._setup_paths()

    def __getattr__(self, name):
        """便于访问配置项"""
        return getattr(self.cfg, name)

    def __getitem__(self, key):
        return self.cfg[key]
