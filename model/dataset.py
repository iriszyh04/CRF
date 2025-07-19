import torch
from torch.utils.data import Dataset, DataLoader
from utils import commonUtils


class NerDataset(Dataset):
    def __init__(self, features):
        """
        初始化 NerDataset
        :param features: List of BertFeature instances, each containing token_ids, attention_masks, token_type_ids, and labels
        """
        self.nums = len(features)

        # 将 features 转化为 tensor 并存储
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        """
        返回数据集的大小
        """
        return self.nums

    def __getitem__(self, index):
        """
        获取一个样本的数据
        :param index: 样本的索引
        :return: 返回 token_ids, attention_masks, token_type_ids 和 labels
        """
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['labels'] = self.labels[index]

        return data

# 示例：如何使用这个 Dataset
# 假设你已经有了一个 features 列表，里面的每一项都是 BertFeature 实例
# features = [BertFeature(...), BertFeature(...), ...]

# 创建 NerDataset 实例
# dataset = NerDataset(features)

# 使用 DataLoader 加载数据
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
