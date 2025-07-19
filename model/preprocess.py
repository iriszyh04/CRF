import os
import json
import logging
from transformers import AlbertTokenizer  # 修改为ALBERT的Tokenizer
from utils import cutSentences, commonUtils

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels

class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # ALBERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids

class AlbertFeature(BaseFeature):  # 修改为AlbertFeature
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(AlbertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels

class NerProcessor:
    def __init__(self, cut_sent=True, cut_sent_len=256):
        self.cut_sent = cut_sent
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        for i, item in enumerate(raw_examples):
            text = item['text']
            if self.cut_sent:
                sentences = cutSentences.cut_sent_for_bert(text, self.cut_sent_len)
                start_index = 0

                for sent in sentences:
                    labels = cutSentences.refactor_labels(sent, item['labels'], start_index)

                    start_index += len(sent)

                    examples.append(InputExample(set_type=set_type,
                                                 text=sent,
                                                 labels=labels))
            else:
                labels = item['labels']
                if len(labels) != 0:
                    labels = [(label[1],label[4],label[2]) for label in labels]
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             labels=labels))
        return examples

def convert_bert_example(ex_idx, example: InputExample, tokenizer: AlbertTokenizer,  # 修改为AlbertTokenizer
                         max_seq_len, ent2id, labels):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    callback_info = (raw_text,)
    callback_labels = {x: [] for x in labels}
    
    for _label in entities:
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info += (callback_labels,)
    
    tokens = [i for i in raw_text]

    label_ids = [0] * len(tokens)

    for ent in entities:
        ent_type = ent[0]  # 类别
        ent_start = ent[-1]  # 起始位置
        ent_end = ent_start + len(ent[1]) - 1

        if ent_start == ent_end:
            label_ids[ent_start] = ent2id['S-' + ent_type]
        else:
            label_ids[ent_start] = ent2id['B-' + ent_type]
            label_ids[ent_end] = ent2id['E-' + ent_type]
            for i in range(ent_start + 1, ent_end):
                label_ids[i] = ent2id['I-' + ent_type]

    if len(label_ids) > max_seq_len - 2:
        label_ids = label_ids[:max_seq_len - 2]

    label_ids = [0] + label_ids + [0]

    if len(label_ids) < max_seq_len:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length  # PAD label都为O

    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation='longest_first',
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    feature = AlbertFeature(  # 修改为AlbertFeature
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, albert_dir, ent2id, labels):  # 修改为albert_dir
    tokenizer = AlbertTokenizer(os.path.join(albert_dir, 'vocab.txt'))  # 修改为AlbertTokenizer
    features = []
    callback_info = []

    for i, example in enumerate(examples):
        if not example.text:
          continue
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            ent2id=ent2id,
            tokenizer=tokenizer,
            labels=labels,
        )
        if feature is None:
            continue
        features.append(feature)
        callback_info.append(tmp_callback)

    out = (features,)
    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_data(processor, raw_data_path, json_file, mode, ent2id, labels, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.albert_dir, ent2id, labels)
    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

