import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from albert_base_model import BaseModel
from transformers import AutoConfig, AutoModel, BertModel
from torchcrf import CRF
import config


from transformers import AlbertTokenizer, AlbertForTokenClassification, AlbertConfig
import torch
import torch.nn as nn
from torchcrf import CRF

import os
import torch.nn as nn
from transformers import AutoModel

class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, model_name=""):
        super(BaseModel, self).__init__()

        # 配置文件路径
        config_path = os.path.join(bert_dir, 'config.json')

        # 检查模型路径和配置文件是否存在
        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained model file does not exist'

        # 如果模型名称包含 'albert'，使用 AutoModel 加载 ALBERT 或 ELECTRA 模型
        if 'electra' in model_name or 'albert' in model_name:
            self.bert_module = AutoModel.from_pretrained(bert_dir, output_hidden_states=True,
                                                         hidden_dropout_prob=dropout_prob)
        else:
            # 默认加载 BERT 模型
            self.bert_module = AutoModel.from_pretrained(bert_dir, output_hidden_states=True,
                                                         hidden_dropout_prob=dropout_prob)
        
        # 获取模型配置
        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 BERT / ALBERT 进行一样的初始化
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

class LayerNorm(nn.Module):
    def __init__(self, filters, elementwise_affine=False):
        super(LayerNorm, self).__init__()
        self.LN = nn.LayerNorm([filters],elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.LN(x)
        return out.permute(0, 2, 1)



class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}
        ]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(filters) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(filters) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output




class AlbertNerModel(BaseModel):
    def __init__(self, args, **kwargs):
        # 初始化ALBERT模型，加载ALBERT的配置文件
        super(AlbertNerModel, self).__init__(bert_dir=args.albert_dir, dropout_prob=args.dropout_prob, model_name=args.model_name)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        # 加载ALBERT配置文件，获取嵌入维度
        self.albert_config = AlbertConfig.from_pretrained(args.albert_dir)
        out_dims = self.albert_config.hidden_size

        # 初始化各种层
        init_blocks = []
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
        init_blocks.append(self.linear)

        # 使用LSTM层，IDCNN层，或中间线性层进行特征提取
        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
        elif args.use_idcnn == "True":
            self.idcnn = IDCNN(out_dims, args.lstm_hidden * 2)
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            out_dims = mid_linear_dims
            self.classifier = nn.Linear(out_dims, args.num_tags)

        # 如果使用CRF层，初始化CRF
        if args.use_crf == 'True':
            self.crf = CRF(args.num_tags, batch_first=True)

        # 初始化权重
        self._init_weights(init_blocks, initializer_range=self.albert_config.initializer_range)

        # 加载ALBERT模型
        self.albert_module = AlbertForTokenClassification.from_pretrained(args.albert_dir, num_labels=args.num_tags)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self, token_ids, attention_masks, token_type_ids, labels):
        # 使用ALBERT模型提取特征
        albert_outputs = self.albert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = albert_outputs[0]  # ALBERT输出的最后一层隐状态 [batch_size, seq_len, hidden_size]
        batch_size = seq_out.size(0)

        # 根据需要使用LSTM/IDCNN或中间线性层
        if self.args.use_lstm == 'True':
            seq_out, _ = self.lstm(seq_out)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1)
        elif self.args.use_idcnn == "True":
            seq_out = self.idcnn(seq_out)
            seq_out = self.linear(seq_out)
        else:
            seq_out = self.mid_linear(seq_out)
            seq_out = self.classifier(seq_out)

        # 如果使用CRF层，进行解码
        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out, mask=attention_masks)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss, ) + (logits,)
            return outputs
        else:
            logits = seq_out
            if labels is None:
                return logits
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            if self.args.use_kd == "True":
                active_loss = attention_masks.view(-1) == 1
                active_logits = seq_out.view(-1, seq_out.size()[2])[active_loss]
                outputs = (loss,) + (active_logits,)
                return outputs
            outputs = (loss,) + (logits,)
            return outputs

if __name__ == '__main__':
    # 获取配置参数
    args = config.Args().get_parser()
    
    # 设置必要的参数
    args.num_tags = 33  # 设置标签的数量
    args.use_lstm = 'True'  # 是否使用 LSTM 层
    args.use_crf = 'True'  # 是否使用 CRF 层
    
    # 使用 ALBERT 代替 BERT
    args.albert_dir = './path_to_your_albert_model'  # 指定 ALBERT 模型路径
    
    # 实例化模型，使用 `AlbertNerModel` 替换 `BertNerModel`
    model = AlbertNerModel(args)
    
    # 打印模型中的所有参数名称
    for name, weight in model.named_parameters():
        print(name)
