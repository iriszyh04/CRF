#!/usr/bin/env python3
"""
交互式NER实体识别脚本
基于 test_performance_evaluation.py 的代码逻辑
用户可以输入任意句子，系统会预测实体类别、位置和对应的文本
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from models.bert_for_ner import BertCrfForNer
from processors.ner_seq import convert_examples_to_features, CnerProcessor
from processors.utils_ner import get_entities
from processors.ner_seq import InputExample

# 导入 safetensors
try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("警告: safetensors 未安装，请运行: pip install safetensors")

class InteractiveNER:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.processor = None
        self.label_list = None
        self.id2label = None
        
        # 实体类型映射
        self.entity_map = {
            'CONT': '联系方式', 'EDU': '教育背景', 'LOC': '地址',
            'NAME': '姓名', 'ORG': '组织机构', 'PRO': '专业',
            'RACE': '民族', 'TITLE': '职位'
        }
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型和tokenizer"""
        print(f"正在加载模型: {self.model_path}")
        
        # 加载配置和tokenizer
        config = BertConfig.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        # 创建模型实例
        self.model = BertCrfForNer(config)
        
        # 加载 safetensors 格式的模型
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print("加载 model.safetensors...")
            state_dict = load_file(safetensors_path)
            self.model.load_state_dict(state_dict)
            print("模型加载成功")
        else:
            print("错误: 未找到 model.safetensors 或 safetensors 库未安装")
            return False
        
        # 设置为评估模式
        self.model.eval()
        
        # 检查设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 加载处理器
        self.processor = CnerProcessor()
        self.label_list = self.processor.get_labels()
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        print(f"模型已加载到: {self.device}")
        print("支持的实体类型:", list(self.entity_map.values()))
        return True
    
    def text_to_chars(self, text):
        """将文本转换为字符列表，处理中文字符"""
        return list(text.strip())
    
    def predict_text(self, text):
        """对输入文本进行实体识别预测"""
        if not self.model:
            print("模型未正确加载")
            return None
        
        # 将文本转换为字符列表
        chars = self.text_to_chars(text)
        
        # 创建示例对象
        example = InputExample(
            guid="interactive-1",
            text_a=chars,
            labels=["O"] * len(chars)  # 占位标签
        )
        
        # 转换为features
        features = convert_examples_to_features(
            examples=[example],
            tokenizer=self.tokenizer,
            label_list=self.label_list,
            max_seq_length=512,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        
        if not features:
            print("文本处理失败")
            return None
        
        feature = features[0]
        
        # 创建张量
        input_ids = torch.tensor([feature.input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([feature.input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "labels": None
            }
            if segment_ids is not None:
                inputs["token_type_ids"] = segment_ids
            
            outputs = self.model(**inputs)
            logits = outputs[0]
            tags = self.model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        
        # 处理预测结果
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP] 去掉首尾
        
        # 转换为标签名称
        pred_labels = [self.id2label[x] for x in preds]
        
        # 确保标签长度与字符长度一致
        if len(pred_labels) > len(chars):
            pred_labels = pred_labels[:len(chars)]
        elif len(pred_labels) < len(chars):
            pred_labels.extend(["O"] * (len(chars) - len(pred_labels)))
        
        # 提取实体
        raw_entities = get_entities(preds, self.id2label, 'bios')
        
        # 转换实体格式为前端期望的格式
        formatted_entities = []
        for entity in raw_entities:
            entity_type = entity[0]  # 实体类型，如 'NAME', 'ORG' 等
            start_idx = entity[1]    # 开始位置
            end_idx = entity[2]      # 结束位置
            
            # 提取实体文本
            entity_text = ''.join(chars[start_idx:end_idx+1])
            
            # 格式化实体信息
            formatted_entity = {
                'type': entity_type,
                'type_name': self.entity_map.get(entity_type, entity_type),
                'text': entity_text,
                'start': start_idx,
                'end': end_idx
            }
            formatted_entities.append(formatted_entity)
        
        return {
            'text': text,
            'chars': chars,
            'labels': pred_labels,
            'entities': formatted_entities
        }
    
    def get_entities_from_bio(self, bio_labels):
        """从BIO/BIOS标签序列中提取实体，兼容S-标签"""
        entities = []
        current_entity = None
        current_start = None
        
        for i, label in enumerate(bio_labels):
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity is not None:
                    entities.append((current_entity, current_start, i-1))
                # 开始新实体
                current_entity = label[2:]
                current_start = i
            elif label.startswith('I-'):
                # 继续当前实体
                if current_entity is None or label[2:] != current_entity:
                    # 标签序列有错误，重新开始
                    current_entity = label[2:]
                    current_start = i
            elif label.startswith('S-'):
                # 单字实体 - BIOS格式特有
                if current_entity is not None:
                    entities.append((current_entity, current_start, i-1))
                entities.append((label[2:], i, i))
                current_entity = None
                current_start = None
            else:  # O标签
                if current_entity is not None:
                    entities.append((current_entity, current_start, i-1))
                current_entity = None
                current_start = None
        
        # 处理最后一个实体
        if current_entity is not None:
            entities.append((current_entity, current_start, len(bio_labels)-1))
        
        return entities
    
    def format_results(self, result):
        """格式化并显示预测结果"""
        if not result:
            return
        
        text = result['text']
        chars = result['chars']
        labels = result['labels']
        entities = result['entities']
        
        print("\n" + "="*80)
        print("NER实体识别结果")
        print("="*80)
        
        print(f"\n输入文本: {text}")
        print(f"字符总数: {len(chars)}")
        
        # 显示字符和对应标签
        print(f"\n字符标注:")
        print("字符: " + " ".join(f"{char:>2}" for char in chars))
        print("标签: " + " ".join(f"{label:>2}" for label in labels))
        
        # 显示识别的实体
        print(f"\n识别到的实体:")
        if entities:
            print("-" * 60)
            print(f"{'实体类型':<12} {'实体文本':<20} {'位置':<10} {'置信度'}")
            print("-" * 60)
            
            for entity in entities:
                entity_type, start, end = entity[0], entity[1], entity[2]
                
                # 提取实体文本
                if start < len(chars) and end < len(chars):
                    entity_text = "".join(chars[start:end+1])
                else:
                    entity_text = f"位置错误({start}-{end})"
                
                # 获取中文实体类型名称
                entity_name = self.entity_map.get(entity_type, entity_type)
                
                print(f"{entity_name:<12} {entity_text:<20} {start}-{end:<8} 高")
        else:
            print("  未识别到任何实体")
        
        print("\n" + "="*80)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*50)
        print("交互式NER实体识别系统")
        print("="*50)
        print("输入中文句子进行实体识别")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看支持的实体类型")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n请输入句子: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                elif user_input.lower() in ['help', '帮助']:
                    print("\n支持的实体类型:")
                    for eng, chn in self.entity_map.items():
                        print(f"  {chn} ({eng})")
                    continue
                elif not user_input:
                    print("请输入有效的句子")
                    continue
                
                # 进行预测
                print(f"\n正在识别实体...")
                result = self.predict_text(user_input)
                
                # 显示结果
                self.format_results(result)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理过程中发生错误: {e}")

def main():
    print("BERT-NER-CRF 交互式实体识别系统")
    print("=" * 50)
    
    # 模型路径
    model_path = "outputs\\cner_output\\bert"
    
    # 检查模型文件
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"错误: 模型文件不存在 {os.path.join(model_path, 'model.safetensors')}")
        return
    
    # 创建NER系统
    ner_system = InteractiveNER(model_path)
    
    # 测试示例
    print("\n先用示例测试系统:")
    test_examples = [
        "我叫张三，住在北京市朝阳区，在清华大学工作。",
        "王小明是软件工程师，他的电话是13800138000。",
        "李教授在厦门泛华集团担任技术总监。"
    ]
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n示例 {i}: {example}")
        result = ner_system.predict_text(example)
        ner_system.format_results(result)
    
    # 进入交互模式
    ner_system.interactive_mode()

if __name__ == "__main__":
    main()