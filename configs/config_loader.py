import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 展平配置字典以兼容现有代码
    flat_config = _flatten_config(config)
    return flat_config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将嵌套的配置字典展平为一层，以兼容现有代码

    Args:
        config: 嵌套的配置字典

    Returns:
        展平后的配置字典
    """
    flat_config = {}

    # 处理模型配置
    if 'model' in config:
        flat_config.update(config['model'])

    # 处理训练配置
    if 'training' in config:
        flat_config.update(config['training'])

    # 处理损失权重
    if 'loss' in config:
        flat_config.update({f'lambda_{k}': v for k, v in config['loss'].items()})

    # 处理数据配置
    if 'data' in config:
        flat_config['data_config'] = config['data']

    # 处理检查点配置
    if 'checkpoint' in config:
        flat_config.update(config['checkpoint'])

    # 处理日志配置
    if 'logging' in config:
        flat_config.update(config['logging'])

    # 处理设备配置
    if 'device' in config:
        flat_config['device'] = config['device']

    return flat_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典，override_config 会覆盖 base_config 中的值

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged
