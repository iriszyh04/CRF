import os
import torch.nn as nn
from transformers import AutoModel  # 只需要导入 AutoModel

class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, model_name=""):
        super(BaseModel, self).__init__()
        current_dir = os.path.dirname(__file__)  # 获取当前文件(BaseModel.py)的目录
        config_path = os.path.join(current_dir, '../albert_model/config.json')  # 构建相对路径

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained model file does not exist'

        # 使用 AutoModel 来自动选择 BERT 或 ALBERT 或其他模型
        self.bert_module = AutoModel.from_pretrained(bert_dir, output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)
        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与模型进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

