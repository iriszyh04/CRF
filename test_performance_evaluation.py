#!/usr/bin/env python3
"""
测试集模型性能评估脚本
使用训练好的 model.safetensors 模型对测试集进行完整评估
包含混淆矩阵和详细性能报告
参考 predict_samples.py 的代码逻辑
"""

import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from models.bert_for_ner import BertCrfForNer
from processors.ner_seq import convert_examples_to_features, CnerProcessor
from processors.utils_ner import get_entities

# 导入 safetensors
try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("警告: safetensors 未安装，请运行: pip install safetensors")

def load_model_and_tokenizer(model_path):
    """加载训练好的模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    
    # 加载配置和tokenizer
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    
    # 创建模型实例
    model = BertCrfForNer(config)
    
    # 加载 safetensors 格式的模型
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
        print("加载 model.safetensors...")
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print("模型加载成功")
    else:
        print("错误: 未找到 model.safetensors 或 safetensors 库未安装")
        return None, None, None
    
    # 设置为评估模式
    model.eval()
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"模型已加载到: {device}")
    return model, tokenizer, device

def predict_test_set(model, tokenizer, device):
    """预测整个测试集"""
    print("\n正在预测测试集...")
    
    # 加载处理器
    processor = CnerProcessor()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 加载测试集数据
    data_dir = "datasets\\cner"
    examples = processor.get_test_examples(data_dir)
    print(f"测试集样本总数: {len(examples)}")
    
    # 转换为features
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=label_list,
        max_seq_length=512,
        cls_token_at_end=False,
        pad_on_left=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    
    # 创建数据加载器
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    
    # 预测
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step % 100 == 0:
                print(f"已处理 {step}/{len(examples)} 样本")
                
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0], 
                "attention_mask": batch[1], 
                "labels": None
            }
            if batch[2] is not None:
                inputs["token_type_ids"] = batch[2]
            
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
            
            # 处理预测结果，参考原代码
            preds = tags[0][1:-1]  # [CLS]XXXX[SEP] 去掉首尾
            label_entities = get_entities(preds, id2label, 'bios')
            
            # 转换为标签名称序列
            pred_label_names = [id2label[x] for x in preds]
            predictions.append(pred_label_names)
    
    print(f"预测完成，共处理 {len(predictions)} 个样本")
    return predictions, examples

def get_entities_from_bio(bio_labels):
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

def evaluate_performance(predictions, examples):
    """评估模型性能"""
    print("\n" + "="*80)
    print("测试集性能评估")
    print("="*80)
    
    # 实体类型映射
    entity_map = {
        'CONT': '联系方式', 'EDU': '教育背景', 'LOC': '地址',
        'NAME': '姓名', 'ORG': '组织机构', 'PRO': '专业',
        'RACE': '民族', 'TITLE': '职位'
    }
    
    # 统计变量
    total_true_entities = 0
    total_pred_entities = 0
    total_correct_entities = 0
    entity_stats = defaultdict(lambda: {'true': 0, 'pred': 0, 'correct': 0})
    
    # 收集所有标签用于混淆矩阵
    all_true_labels = []
    all_pred_labels = []
    
    for pred_seq, example in zip(predictions, examples):
        true_seq = example.labels
        
        # 确保序列长度一致
        min_len = min(len(true_seq), len(pred_seq))
        true_seq = true_seq[:min_len]
        pred_seq = pred_seq[:min_len]
        
        # 添加到混淆矩阵数据
        all_true_labels.extend(true_seq)
        all_pred_labels.extend(pred_seq)
        
        # 提取实体
        true_entities = get_entities_from_bio(true_seq)
        pred_entities = get_entities_from_bio(pred_seq)
        
        total_true_entities += len(true_entities)
        total_pred_entities += len(pred_entities)
        
        # 统计各类别实体
        for entity in true_entities:
            entity_type = entity[0]
            entity_stats[entity_type]['true'] += 1
        
        for entity in pred_entities:
            entity_type = entity[0]
            entity_stats[entity_type]['pred'] += 1
        
        # 统计正确预测的实体
        true_entities_set = set(true_entities)
        pred_entities_set = set(pred_entities)
        correct_entities = true_entities_set & pred_entities_set
        total_correct_entities += len(correct_entities)
        
        for entity in correct_entities:
            entity_type = entity[0]
            entity_stats[entity_type]['correct'] += 1
    
    # 计算整体指标
    overall_precision = total_correct_entities / total_pred_entities if total_pred_entities > 0 else 0
    overall_recall = total_correct_entities / total_true_entities if total_true_entities > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 打印整体结果
    print(f"整体性能指标:")
    print(f"准确率 (Precision): {overall_precision:.4f}")
    print(f"召回率 (Recall): {overall_recall:.4f}")
    print(f"F1分数: {overall_f1:.4f}")
    print(f"真实实体总数: {total_true_entities}")
    print(f"预测实体总数: {total_pred_entities}")
    print(f"正确预测实体数: {total_correct_entities}")
    
    # 各实体类型详细结果
    print("\n各实体类型详细结果:")
    print("-" * 80)
    print(f"{'实体类型':<12} {'准确率':<8} {'召回率':<8} {'F1分数':<8} {'真实数':<6} {'预测数':<6} {'正确数':<6}")
    print("-" * 80)
    
    for entity_type in sorted(entity_stats.keys()):
        if entity_type in entity_map:
            stats = entity_stats[entity_type]
            
            precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
            recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{entity_map[entity_type]:<12} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} "
                  f"{stats['true']:<6} {stats['pred']:<6} {stats['correct']:<6}")
    
    return all_true_labels, all_pred_labels, entity_stats, overall_precision, overall_recall, overall_f1

def create_confusion_matrix(true_labels, pred_labels):
    """创建并保存混淆矩阵"""
    print("\n生成混淆矩阵...")
    
    # 设置中文字体支持
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 获取所有唯一标签
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # 过滤掉一些不重要的标签以便可视化
    important_labels = [label for label in unique_labels if not label.startswith('X') and label != '[START]' and label != '[END]']
    
    # 只保留重要标签的数据
    filtered_true = []
    filtered_pred = []
    for true, pred in zip(true_labels, pred_labels):
        if true in important_labels and pred in important_labels:
            filtered_true.append(true)
            filtered_pred.append(pred)
    
    if not filtered_true:
        print("警告: 没有足够的数据生成混淆矩阵")
        return
    
    # 生成混淆矩阵
    cm = confusion_matrix(filtered_true, filtered_pred, labels=important_labels)
    
    # 创建图表
    plt.figure(figsize=(15, 12))
    
    # 创建热力图
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=important_labels, yticklabels=important_labels,
                     cbar_kws={'label': 'Count'})
    
    # 设置标题和标签，使用Unicode编码确保中文显示
    plt.title('测试集标签预测混淆矩阵\nBERT-NER-CRF Confusion Matrix', 
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('预测标签 (Predicted Labels)', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签 (True Labels)', fontsize=14, fontweight='bold')
    
    # 旋转标签以便更好显示
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表，使用多种格式
    output_path = "outputs\\cner_output\\bert\\confusion_matrix.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"混淆矩阵已保存到: {output_path}")
        
        # 同时保存PDF格式以确保中文显示
        pdf_path = "outputs\\cner_output\\bert\\confusion_matrix.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"PDF版本已保存到: {pdf_path}")
        
    except Exception as e:
        print(f"保存混淆矩阵时出错: {e}")
    
    plt.show()
    
    # 生成分类报告
    try:
        report = classification_report(filtered_true, filtered_pred, 
                                     target_names=important_labels, 
                                     output_dict=True, zero_division=0)
        
        # 保存详细报告
        report_path = "outputs\\cner_output\\bert\\classification_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"详细分类报告已保存到: {report_path}")
        
        # 打印简要分类报告
        print("\n分类报告摘要:")
        print(f"{'标签':<15} {'准确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
        print("-" * 60)
        
        for label in important_labels:
            if label in report:
                precision = report[label]['precision']
                recall = report[label]['recall']
                f1 = report[label]['f1-score']
                support = report[label]['support']
                print(f"{label:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
        
        return cm, report
        
    except Exception as e:
        print(f"生成分类报告时出错: {e}")
        return cm, None

def save_performance_report(entity_stats, overall_metrics, predictions, examples):
    """保存性能评估报告"""
    print("\n保存性能评估报告...")
    
    report = {
        "overall_metrics": {
            "precision": overall_metrics[0],
            "recall": overall_metrics[1], 
            "f1_score": overall_metrics[2],
            "total_samples": len(examples)
        },
        "entity_metrics": {},
        "sample_count": len(predictions)
    }
    
    # 实体类型映射
    entity_map = {
        'CONT': '联系方式', 'EDU': '教育背景', 'LOC': '地址',
        'NAME': '姓名', 'ORG': '组织机构', 'PRO': '专业',
        'RACE': '民族', 'TITLE': '职位'
    }
    
    for entity_type, stats in entity_stats.items():
        if entity_type in entity_map:
            precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
            recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            report["entity_metrics"][entity_type] = {
                "entity_name": entity_map[entity_type],
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_count": stats['true'],
                "pred_count": stats['pred'],
                "correct_count": stats['correct']
            }
    
    # 保存报告
    report_path = "outputs\\cner_output\\bert\\test_performance_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存文本版本报告
    text_report_path = "outputs\\cner_output\\bert\\test_performance_report.txt"
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("BERT-NER-CRF 测试集性能评估报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("整体性能指标:\n")
        f.write(f"准确率: {overall_metrics[0]:.4f}\n")
        f.write(f"召回率: {overall_metrics[1]:.4f}\n")
        f.write(f"F1分数: {overall_metrics[2]:.4f}\n")
        f.write(f"测试样本数: {len(examples)}\n\n")
        
        f.write("各实体类型详细结果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'实体类型':<12} {'准确率':<8} {'召回率':<8} {'F1分数':<8} {'真实数':<6} {'预测数':<6} {'正确数':<6}\n")
        f.write("-" * 80 + "\n")
        
        for entity_type in sorted(entity_stats.keys()):
            if entity_type in entity_map:
                stats = entity_stats[entity_type]
                precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
                recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{entity_map[entity_type]:<12} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} "
                       f"{stats['true']:<6} {stats['pred']:<6} {stats['correct']:<6}\n")
    
    print(f"性能报告已保存到:")
    print(f"  JSON格式: {report_path}")
    print(f"  文本格式: {text_report_path}")

def main():
    print("BERT-NER-CRF 测试集性能评估脚本")
    print("使用训练好的 model.safetensors 模型")
    print("=" * 50)
    
    # 模型路径
    model_path = "outputs\\cner_output\\bert"
    
    # 检查模型文件
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"错误: 模型文件不存在 {os.path.join(model_path, 'model.safetensors')}")
        return
    
    # 加载模型
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None:
        print("模型加载失败")
        return
    
    # 预测测试集
    predictions, examples = predict_test_set(model, tokenizer, device)
    
    # 评估性能
    true_labels, pred_labels, entity_stats, precision, recall, f1 = evaluate_performance(predictions, examples)
    
    # 生成混淆矩阵
    try:
        create_confusion_matrix(true_labels, pred_labels)
    except Exception as e:
        print(f"生成混淆矩阵时出错: {e}")
        print("可能需要安装: pip install matplotlib seaborn")
    
    # 保存报告
    save_performance_report(entity_stats, (precision, recall, f1), predictions, examples)
    
    print("\n" + "="*50)
    print("测试集性能评估完成！")
    print("生成的文件:")
    print("  - outputs\\cner_output\\bert\\confusion_matrix.png")
    print("  - outputs\\cner_output\\bert\\classification_report.json")
    print("  - outputs\\cner_output\\bert\\test_performance_report.json")
    print("  - outputs\\cner_output\\bert\\test_performance_report.txt")
    print("="*50)

if __name__ == "__main__":
    main()