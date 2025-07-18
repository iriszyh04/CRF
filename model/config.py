import argparse
import os
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/', help='the output dir for model checkpoints')

        # 修改为 ALBERT 的本地路径
        parser.add_argument('--bert_dir', default='C:/Users/Administrator/代码项目/albert预训练模型', help='albert dir for uer')

        parser.add_argument('--data_dir', default='./data/cner/', help='data dir for uer')
        parser.add_argument('--log_dir', default='./logs/', help='log dir for uer')

        # other args
        parser.add_argument('--num_tags', default=53, type=int, help='number of tags')
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=256, type=int)
        parser.add_argument('--eval_batch_size', default=12, type=int)
        parser.add_argument('--swa_start', default=3, type=int, help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=15, type=int, help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='drop out probability')

        parser.add_argument('--lr', default=3e-5, type=float, help='albert学习率')
        parser.add_argument('--other_lr', default=3e-4, type=float, help='bilstm和多层感知机学习率')
        parser.add_argument('--crf_lr', default=3e-2, type=float, help='条件随机场学习率')

        parser.add_argument('--max_grad_norm', default=1, type=float, help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--use_lstm', type=str, default='True', help='是否使用BiLstm')
        parser.add_argument('--lstm_hidden', default=128, type=int, help='lstm隐藏层大小')
        parser.add_argument('--num_layers', default=1, type=int, help='lstm层数大小')
        parser.add_argument('--dropout', default=0.3, type=float, help='lstm中dropout的设置')
        parser.add_argument('--use_crf', type=str, default='True', help='是否使用Crf')
        parser.add_argument('--use_idcnn', type=str, default='False', help='是否使用Idcnn')
        parser.add_argument('--data_name', type=str, default='c', help='数据集名字')
        parser.add_argument('--model_name', type=str, default='albert', help='模型名字')
        parser.add_argument('--use_tensorboard', type=str, default='True', help='是否使用tensorboard可视化')
        parser.add_argument('--use_kd', type=str, default='False', help='是否使用知识蒸馏')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

# ALBERT 模型的基类
class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, model_name=""):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), 'pretrained albert file does not exist'

        # 加载 ALBERT 模型
        self.bert_module = AutoModel.from_pretrained(bert_dir, output_hidden_states=True, hidden_dropout_prob=dropout_prob)
        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
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

