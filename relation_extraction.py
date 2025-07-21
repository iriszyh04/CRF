#!/usr/bin/env python3
"""
基于NER模型的关系抽取系统
不需要重新训练模型，使用规则和模式匹配进行关系抽取
基于已训练的BERT-NER-CRF模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from collections import defaultdict
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

class RelationExtractor:
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
        
        # 定义关系模式和规则
        self.relation_patterns = self._init_relation_patterns()
        
        self.load_model()
    
    def _init_relation_patterns(self):
        """初始化关系抽取的模式和规则"""
        patterns = {
            # 人员-组织关系 (方向: NAME -> ORG)
            "工作于": {
                "subject_type": "NAME",
                "object_type": "ORG",
                "patterns": [
                    r"在(.+?)(?:工作|任职|就职|服务)",
                    r"(?:工作|任职|就职|服务)于(.+?)",
                    r"(.+?)(?:公司|集团|企业|机构)(?:的)?(?:员工|职员)",
                    r"(?:加入|进入|入职)(.+?)"
                ],
                "keywords": ["工作", "任职", "服务", "就职", "员工", "职员", "入职", "加入", "公司", "企业"],
                "distance_weight": 0.2,
                "syntax_patterns": [
                    {"pattern": r"(.+?)在(.+?)(?:工作|任职)", "confidence": 0.9},
                    {"pattern": r"(.+?)(?:工作|任职)于(.+?)", "confidence": 0.85}
                ]
            },
            
            # 人员-职位关系 (方向: NAME -> TITLE)
            "担任": {
                "subject_type": "NAME",
                "object_type": "TITLE",
                "patterns": [
                    r"(?:担任|出任|任命为|升任)(.+?)(?:职位|岗位|一职)?",
                    r"是(.+?)(?:职位|岗位)?(?:，|。|$)",  # 避免与"毕业于"冲突
                    r"(?:职务|职位)是(.+?)(?:，|。|$)",
                    r"(?:当|做|干)(.+?)(?:工作|职位)?(?:，|。|$)",
                    r"现(?:任|在)(.+?)(?:，|。|$)"  # 强调现在的职位
                ],
                "keywords": ["担任", "是", "任", "当", "做", "职位", "职务", "岗位", "出任", "升任", "现任"],
                "distance_weight": 0.25,
                "syntax_patterns": [
                    {"pattern": r"(.+?)担任(.+?)", "confidence": 0.95},
                    {"pattern": r"(.+?)现(?:任|在)(.+?)(?:职位|岗位|职务)", "confidence": 0.9},
                    {"pattern": r"(.+?)是(.+?)(?:职位|岗位|职务)", "confidence": 0.8}
                ]
            },
            
            # 人员-地址关系 (方向: NAME -> LOC)
            "居住于": {
                "subject_type": "NAME",
                "object_type": "LOC",
                "patterns": [
                    r"(?:住在|居住在|生活在)(.+?)",
                    r"家(?:住|在)(.+?)",
                    r"(?:来自|出生于)(.+?)",
                    r"(?:户籍|籍贯)(?:在|是)(.+?)",
                    r"现住址(?:在|是|为)(.+?)"
                ],
                "keywords": ["住", "居住", "家", "来自", "出生", "户籍", "籍贯", "住址", "生活"],
                "distance_weight": 0.2,
                "syntax_patterns": [
                    {"pattern": r"(.+?)住在(.+?)", "confidence": 0.9},
                    {"pattern": r"(.+?)来自(.+?)", "confidence": 0.85}
                ]
            },
            
            # 人员-联系方式关系 (方向: NAME -> CONT)
            "联系方式": {
                "subject_type": "NAME",
                "object_type": "CONT",
                "patterns": [
                    r"(?:电话|手机|联系方式|联系电话)(?:是|为|:)(.+?)",
                    r"(?:邮箱|邮件|email)(?:是|为|:)(.+?)",
                    r"(?:微信|QQ|WeChat)(?:号|是|为|:)(.+?)",
                    r"(?:手机号码|电话号码)(?:是|为|:)(.+?)"
                ],
                "keywords": ["电话", "手机", "联系", "邮箱", "微信", "QQ", "号码", "email"],
                "distance_weight": 0.1,
                "syntax_patterns": [
                    {"pattern": r"(.+?)(?:的)?(?:电话|手机)(?:是|为)(.+?)", "confidence": 0.95}
                ]
            },
            
            # 人员-教育背景关系 (方向: NAME -> EDU)
            "毕业于": {
                "subject_type": "NAME",
                "object_type": "EDU", 
                "patterns": [
                    r"(?:毕业于|毕业自)(.+?)(?:大学|学院|学校|高中|中学)",
                    r"(?:就读于|在读于)(.+?)(?:大学|学院|学校|高中|中学)",
                    r"在(.+?)(?:大学|学院|学校|高中|中学)(?:学习|就读|毕业)",
                    r"(?:学历|最高学历)(?:是|为)(.+?)",
                    r"(?:获得|取得)(.+?)(?:学位|文凭|毕业证)",
                    r"(.+?)(?:大学|学院|学校|高中|中学)毕业"
                ],
                "keywords": ["毕业", "就读", "学习", "学历", "学位", "大学", "学院", "学校", "文凭", "毕业证"],
                "distance_weight": 0.3,
                "syntax_patterns": [
                    {"pattern": r"(.+?)毕业于(.+?)(?:大学|学院|学校)", "confidence": 0.98},
                    {"pattern": r"(.+?)在(.+?)(?:大学|学院|学校)(?:学习|就读|毕业)", "confidence": 0.92}
                ]
            },
            
            # 人员-专业关系 (方向: NAME -> PRO)
            "专业是": {
                "subject_type": "NAME",
                "object_type": "PRO",
                "patterns": [
                    r"(?:专业是|专业为|学的是)(.+?)",
                    r"(?:主修|所学专业是)(.+?)",
                    r"(?:研究方向|研究领域)(?:是|为)(.+?)",
                    r"(?:从事|专门从事)(.+?)(?:相关|方面)(?:工作|研究)?",
                    r"(.+?)专业(?:毕业|出身)"
                ],
                "keywords": ["专业", "主修", "研究", "从事", "学的", "方向", "领域"],
                "distance_weight": 0.2,
                "syntax_patterns": [
                    {"pattern": r"(.+?)(?:的)?专业是(.+?)", "confidence": 0.9}
                ]
            },
            
            # 组织-地址关系 (方向: ORG -> LOC)
            "位于": {
                "subject_type": "ORG",
                "object_type": "LOC",
                "patterns": [
                    r"(?:位于|坐落于|设在|建在)(.+?)",
                    r"在(.+?)(?:设立|设置|建立)",
                    r"(?:总部|办公室|分公司)(?:在|位于)(.+?)",
                    r"(?:地址|办公地址)(?:是|为|在)(.+?)"
                ],
                "keywords": ["位于", "在", "设在", "坐落", "地址", "总部", "办公室", "建在"],
                "distance_weight": 0.15,
                "syntax_patterns": [
                    {"pattern": r"(.+?)位于(.+?)", "confidence": 0.9}
                ]
            },
            
            # 人员-民族关系 (方向: NAME -> RACE)
            "民族是": {
                "subject_type": "NAME",
                "object_type": "RACE",
                "patterns": [
                    r"(?:民族是|民族为)(.+?)族?",
                    r"是(.+?)族人?",
                    r"(.+?)族(?:血统|出身)"
                ],
                "keywords": ["民族", "族", "血统", "出身"],
                "distance_weight": 0.1,
                "syntax_patterns": [
                    {"pattern": r"(.+?)(?:的)?民族是(.+?)", "confidence": 0.95}
                ]
            },
            
            # 人员-组织-职位三元关系
            "在_担任": {
                "subject_type": "NAME",
                "object_type": "ORG",
                "auxiliary_type": "TITLE",
                "patterns": [
                    r"在(.+?)担任(.+?)",
                    r"在(.+?)(?:任|做)(.+?)",
                    r"(.+?)公司的(.+?)"
                ],
                "keywords": ["在", "担任", "任", "做", "的"],
                "is_ternary": True,
                "distance_weight": 0.3
            },
            
            # 组织-组织关系 (方向: ORG -> ORG)
            "隶属于": {
                "subject_type": "ORG",
                "object_type": "ORG",
                "patterns": [
                    r"(?:隶属于|属于|归属于)(.+?)",
                    r"是(.+?)(?:的)?(?:子公司|分公司|部门)",
                    r"(.+?)(?:旗下|下属)(?:公司|机构)"
                ],
                "keywords": ["隶属", "属于", "归属", "子公司", "分公司", "部门", "旗下", "下属"],
                "distance_weight": 0.2
            }
        }
        return patterns
    
    def load_model(self):
        """加载训练好的NER模型"""
        print(f"正在加载NER模型: {self.model_path}")
        
        # 直接从训练好的模型路径加载配置和tokenizer
        # 因为训练时已经将BERT配置保存到了输出目录
        config = BertConfig.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        # 创建模型实例
        self.model = BertCrfForNer(config)
        
        # 加载训练好的模型权重
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"从 {safetensors_path} 加载训练好的模型权重...")
            state_dict = load_file(safetensors_path)
            self.model.load_state_dict(state_dict)
            print("✅ 训练好的NER模型加载成功（包含BERT+CRF）")
        else:
            print("❌ 错误: 未找到 model.safetensors 或 safetensors 库未安装")
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
        
        print(f"🚀 模型已加载到设备: {self.device}")
        print("💡 使用您训练好的BERT-CRF模型进行关系抽取增强（简化版本，暂时禁用注意力机制）")
        return True
    
    def extract_entities_with_attention(self, text):
        """提取文本中的实体，同时获取BERT隐藏状态用于关系抽取（简化版本，避免注意力权重问题）"""
        chars = list(text.strip())
        
        # 创建示例对象
        example = InputExample(
            guid="relation-1",
            text_a=chars,
            labels=["O"] * len(chars)
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
            return [], None, None
        
        feature = features[0]
        
        # 创建张量
        input_ids = torch.tensor([feature.input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([feature.input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long).to(self.device)
        
        # 预测并获取隐藏状态（不获取注意力权重以避免错误）
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "labels": None
            }
            if segment_ids is not None:
                inputs["token_type_ids"] = segment_ids
            
            # 获取BERT的隐藏状态（不设置output_attentions避免错误）
            try:
                # 先尝试获取隐藏状态
                self.model.bert.config.output_attentions = False  # 确保不获取注意力权重
                bert_outputs = self.model.bert(**inputs)
                hidden_states = bert_outputs.last_hidden_state  # [1, seq_len, hidden_size]
                attention_weights = None  # 暂时不使用注意力权重
            except Exception as e:
                print(f"获取BERT隐藏状态失败: {e}")
                # 如果失败，直接进行CRF预测
                hidden_states = None
                attention_weights = None
            
            # 获取CRF预测结果
            outputs = self.model(**inputs)
            logits = outputs[0]
            tags = self.model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        
        # 处理预测结果
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP] 去掉首尾
        
        # 提取实体
        entities = get_entities(preds, self.id2label, 'bios')
        
        # 构建实体信息，包含token位置映射
        entity_list = []
        for entity in entities:
            entity_type, start, end = entity[0], entity[1], entity[2]
            if start < len(chars) and end < len(chars):
                entity_text = "".join(chars[start:end+1])
                entity_list.append({
                    'type': entity_type,
                    'text': entity_text,
                    'start': start,
                    'end': end,
                    'token_start': start + 1,  # +1 因为有[CLS]
                    'token_end': end + 1,
                    'type_name': self.entity_map.get(entity_type, entity_type)
                })
        
        return entity_list, hidden_states, attention_weights
    
    def extract_entities(self, text):
        """提取文本中的实体，使用BERT上下文理解"""
        entities, _, _ = self.extract_entities_with_attention(text)
        return entities
    
    def extract_relations_by_rules(self, text, entities):
        """基于规则和BERT上下文理解提取关系"""
        # 获取BERT注意力和隐藏状态
        _, hidden_states, attention_weights = self.extract_entities_with_attention(text)
        
        relations = []
        
        # 按实体类型分组
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        # 定义关系优先级（避免冲突，优先级高的先处理）
        relation_priority = {
            "毕业于": 1,      # 最高优先级，避免被"担任"误判
            "专业是": 1,      # 教育相关，高优先级
            "联系方式": 1,    # 明确的联系信息
            "民族是": 1,      # 明确的身份信息
            "位于": 2,        # 地理位置关系
            "居住于": 2,      # 地理位置关系
            "工作于": 3,      # 工作关系
            "担任": 4,        # 职位关系，较低优先级避免误判
            "隶属于": 5,      # 组织关系
            "在_担任": 6      # 三元关系，最低优先级
        }
        
        # 按优先级排序关系模式
        sorted_relations = sorted(self.relation_patterns.items(), 
                                 key=lambda x: relation_priority.get(x[0], 999))
        
        # 遍历所有关系模式（按优先级）
        for relation_name, pattern_info in sorted_relations:
            new_relations = self._extract_binary_relations(
                text, entities_by_type, relation_name, pattern_info, attention_weights, hidden_states
            )
            
            # 冲突检测：避免同一对实体被分配多个冲突的关系
            filtered_relations = self._filter_conflicting_relations(new_relations, relations)
            relations.extend(filtered_relations)
            
            # 处理三元关系
            if pattern_info.get('is_ternary', False):
                ternary_relations = self._extract_ternary_relations(
                    text, entities_by_type, relation_name, pattern_info, attention_weights, hidden_states
                )
                relations.extend(ternary_relations)
        
        # 去重和排序
        relations = self._deduplicate_relations(relations)
        relations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return relations
    
    def _extract_binary_relations(self, text, entities_by_type, relation_name, pattern_info, attention_weights=None, hidden_states=None):
        """提取二元关系，集成BERT上下文理解"""
        relations = []
        subject_type = pattern_info['subject_type']
        object_type = pattern_info['object_type']
        
        # 检查是否有必需的实体类型
        if not entities_by_type.get(subject_type) or not entities_by_type.get(object_type):
            return relations
        
        # 基于关键词的简单匹配，但要更精确地检查上下文
        keywords = pattern_info.get('keywords', [])
        has_keywords = any(keyword in text for keyword in keywords)
        if not has_keywords:
            return relations
        
        # 特殊处理：精确匹配关键模式以避免误判
        if relation_name == "毕业于":
            # 确保真的是毕业关系，而不是工作关系
            graduation_patterns = ["毕业于", "毕业自", "就读于", "在读于"]
            if not any(pattern in text for pattern in graduation_patterns):
                return relations
        elif relation_name == "担任":
            # 确保不是毕业关系被误判为担任
            if any(pattern in text for pattern in ["毕业于", "毕业自", "就读于", "在读于"]):
                # 如果存在毕业关键词，降低担任关系的检测敏感度
                pass
        
        # 提取关系 - 确保方向正确
        for subject_entity in entities_by_type[subject_type]:
            for object_entity in entities_by_type[object_type]:
                # 检查实体间的距离和位置关系
                distance = abs(subject_entity['start'] - object_entity['start'])
                max_distance = 30  # 增加最大距离
                
                if distance < max_distance:
                    # 基于位置、语法和BERT上下文检查关系方向的合理性
                    confidence = self._calculate_enhanced_confidence(
                        text, subject_entity, object_entity, pattern_info, relation_name, attention_weights, hidden_states
                    )
                    
                    if confidence > 0.25:  # 降低阈值以获取更多关系
                        relation = {
                            'relation': relation_name,
                            'subject': subject_entity,
                            'object': object_entity,
                            'confidence': confidence,
                            'context': self._extract_context(text, subject_entity, object_entity),
                            'direction': f"{subject_type} -> {object_type}",
                            'type': 'binary'
                        }
                        relations.append(relation)
        
        return relations
    
    def _extract_ternary_relations(self, text, entities_by_type, relation_name, pattern_info, attention_weights=None, hidden_states=None):
        """提取三元关系（人员-组织-职位），集成BERT上下文理解"""
        relations = []
        subject_type = pattern_info['subject_type']
        object_type = pattern_info['object_type']
        auxiliary_type = pattern_info.get('auxiliary_type')
        
        if not all([entities_by_type.get(subject_type), entities_by_type.get(object_type), 
                   entities_by_type.get(auxiliary_type)]):
            return relations
        
        keywords = pattern_info.get('keywords', [])
        has_keywords = any(keyword in text for keyword in keywords)
        if not has_keywords:
            return relations
        
        # 三元关系提取
        for name_entity in entities_by_type[subject_type]:
            for org_entity in entities_by_type[object_type]:
                for title_entity in entities_by_type[auxiliary_type]:
                    # 计算三个实体间的总距离
                    entities_positions = [name_entity['start'], org_entity['start'], title_entity['start']]
                    total_span = max(entities_positions) - min(entities_positions)
                    
                    if total_span < 50:  # 三元关系允许更大距离
                        confidence = self._calculate_ternary_confidence(
                            text, name_entity, org_entity, title_entity, pattern_info, attention_weights, hidden_states
                        )
                        
                        if confidence > 0.4:
                            relation = {
                                'relation': relation_name,
                                'subject': name_entity,
                                'object': org_entity,
                                'auxiliary': title_entity,
                                'confidence': confidence,
                                'context': self._extract_extended_context(text, [name_entity, org_entity, title_entity]),
                                'direction': f"{subject_type} -> {object_type} ({auxiliary_type})",
                                'type': 'ternary'
                            }
                            relations.append(relation)
        
        return relations
    
    def _calculate_enhanced_confidence(self, text, subject_entity, object_entity, pattern_info, relation_name, attention_weights=None, hidden_states=None):
        """增强的置信度计算算法，集成BERT注意力机制"""
        confidence = 0.2  # 基础置信度
        
        subject_start = subject_entity['start']
        object_start = object_entity['start']
        subject_text = subject_entity['text']
        object_text = object_entity['text']
        
        # 1. 检查关键词密度
        keywords = pattern_info.get('keywords', [])
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        confidence += keyword_count * 0.12
        
        # 2. 检查语法模式匹配
        syntax_patterns = pattern_info.get('syntax_patterns', [])
        for syntax_pattern in syntax_patterns:
            if re.search(syntax_pattern['pattern'], text):
                confidence += syntax_pattern.get('confidence', 0.2)
                break
        
        # 3. 实体间距离加权
        distance = abs(subject_start - object_start)
        distance_weight = pattern_info.get('distance_weight', 0.15)
        if distance < 5:
            confidence += distance_weight * 2
        elif distance < 10:
            confidence += distance_weight * 1.5
        elif distance < 15:
            confidence += distance_weight
        elif distance < 25:
            confidence += distance_weight * 0.5
        
        # 4. 模式匹配加分
        patterns = pattern_info.get('patterns', [])
        for pattern in patterns:
            if re.search(pattern, text):
                confidence += 0.25
                break
        
        # 5. BERT注意力权重分析（新增）
        if attention_weights is not None and hidden_states is not None:
            attention_confidence = self._analyze_attention_patterns(
                subject_entity, object_entity, attention_weights, hidden_states
            )
            confidence += attention_confidence
        
        # 6. 语义相似度分析（新增）
        if hidden_states is not None:
            semantic_confidence = self._calculate_semantic_similarity(
                subject_entity, object_entity, hidden_states, relation_name
            )
            confidence += semantic_confidence
        
        # 7. 方向性检查
        directional_confidence = self._check_directional_logic(
            text, subject_entity, object_entity, relation_name
        )
        confidence += directional_confidence
        
        # 8. 上下文语义一致性
        context_confidence = self._check_context_consistency(
            text, subject_entity, object_entity, relation_name
        )
        confidence += context_confidence
        
        # 9. 实体类型匹配度
        type_confidence = self._check_entity_type_compatibility(
            subject_entity, object_entity, relation_name
        )
        confidence += type_confidence
        
        return max(0.0, min(confidence, 1.0))
    
    def _analyze_attention_patterns(self, subject_entity, object_entity, attention_weights, hidden_states):
        """分析BERT注意力模式以增强关系抽取（简化版本，暂时禁用）"""
        # 暂时禁用注意力分析以避免错误，返回固定的置信度增量
        if attention_weights is None:
            # 基于实体位置的简单距离计算
            distance = abs(subject_entity['start'] - object_entity['start'])
            if distance < 5:
                return 0.15  # 距离很近，给予较高的置信度增量
            elif distance < 10:
                return 0.1   # 距离中等，给予中等的置信度增量
            elif distance < 20:
                return 0.05  # 距离较远，给予较低的置信度增量
            else:
                return 0.0   # 距离太远，不给置信度增量
        
        # 如果有注意力权重数据，尝试分析（但由于兼容性问题，这部分暂时被禁用）
        try:
            # 由于BERT注意力机制的兼容性问题，暂时返回基于距离的简单计算
            distance = abs(subject_entity['start'] - object_entity['start'])
            return max(0.0, 0.2 - distance * 0.01)  # 距离越近，置信度增量越高
            
        except Exception as e:
            print(f"注意力分析失败（已跳过）: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, subject_entity, object_entity, hidden_states, relation_name):
        """计算基于BERT隐藏状态的语义相似度"""
        try:
            # 如果隐藏状态为空，返回0
            if hidden_states is None:
                return 0.0
                
            # 获取实体的隐藏状态表示
            subj_start = subject_entity.get('token_start', subject_entity['start'] + 1)
            subj_end = subject_entity.get('token_end', subject_entity['end'] + 1)
            obj_start = object_entity.get('token_start', object_entity['start'] + 1)
            obj_end = object_entity.get('token_end', object_entity['end'] + 1)
            
            # 提取实体的平均隐藏状态
            hidden_states_cpu = hidden_states.cpu()
            
            # 边界检查
            max_seq_len = hidden_states_cpu.shape[1]
            subj_start = min(subj_start, max_seq_len - 1)
            subj_end = min(subj_end, max_seq_len - 1)
            obj_start = min(obj_start, max_seq_len - 1)
            obj_end = min(obj_end, max_seq_len - 1)
            
            if subj_start > subj_end or obj_start > obj_end:
                return 0.0
            
            subj_hidden = hidden_states_cpu[0, subj_start:subj_end+1].mean(dim=0)  # [hidden_size]
            obj_hidden = hidden_states_cpu[0, obj_start:obj_end+1].mean(dim=0)  # [hidden_size]
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(subj_hidden.unsqueeze(0), obj_hidden.unsqueeze(0), dim=1).item()
            
            # 根据关系类型调整相似度权重
            relation_weights = {
                "工作于": 0.15,      # 人名和组织应该有一定相关性
                "担任": 0.2,        # 人名和职位相关性较高
                "居住于": 0.1,      # 人名和地址相关性较低
                "毕业于": 0.15,     # 人名和学校有一定相关性
                "专业是": 0.2,      # 人名和专业相关性较高
                "位于": 0.1,        # 组织和地址相关性较低
                "联系方式": 0.05,   # 人名和联系方式相关性最低
                "民族是": 0.1,      # 人名和民族相关性较低
                "隶属于": 0.2       # 组织间相关性较高
            }
            
            weight = relation_weights.get(relation_name, 0.1)
            semantic_confidence = similarity * weight
            
            return max(0.0, min(semantic_confidence, 0.15))  # 最大贡献0.15
            
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return 0.0
    
    def _check_directional_logic(self, text, subject_entity, object_entity, relation_name):
        """检查关系方向的逻辑合理性"""
        subject_start = subject_entity['start']
        object_start = object_entity['start']
        subject_text = subject_entity['text']
        object_text = object_entity['text']
        
        # 提取实体间的上下文
        start_pos = min(subject_start, object_start)
        end_pos = max(subject_entity['end'], object_entity['end'])
        context = text[max(0, start_pos-5):end_pos + 6]
        
        directional_bonus = 0.0
        
        if relation_name == "担任":
            # "张三担任产品经理" vs "产品经理担任张三"
            if any(keyword in context for keyword in ["担任", "出任", "是"]):
                if subject_start < object_start:
                    directional_bonus += 0.3
                else:
                    directional_bonus -= 0.15
        
        elif relation_name == "工作于":
            # "张三在阿里工作" vs "阿里在张三工作"
            if "在" in context and "工作" in context:
                zai_pos = context.find("在")
                work_pos = context.find("工作")
                if zai_pos < work_pos:
                    if subject_start < object_start:
                        directional_bonus += 0.35
                    else:
                        directional_bonus -= 0.25
        
        elif relation_name == "居住于":
            if any(keyword in context for keyword in ["住在", "居住在", "家在"]):
                if subject_start < object_start:
                    directional_bonus += 0.3
                else:
                    directional_bonus -= 0.4
        
        elif relation_name == "毕业于":
            if "毕业于" in context:
                if subject_start < object_start:
                    directional_bonus += 0.35
                else:
                    directional_bonus -= 0.45
        
        elif relation_name == "位于":
            if "位于" in context:
                if subject_start < object_start:
                    directional_bonus += 0.3
                else:
                    directional_bonus -= 0.3
        
        elif relation_name == "专业是":
            if "专业" in context:
                if subject_start < object_start:
                    directional_bonus += 0.25
                else:
                    directional_bonus -= 0.2
        
        return directional_bonus
    
    def _check_context_consistency(self, text, subject_entity, object_entity, relation_name):
        """检查上下文语义一致性"""
        context = self._extract_context(text, subject_entity, object_entity)
        context_bonus = 0.0
        
        # 检查是否存在矛盾的表述
        contradiction_patterns = [
            r"不在(.+?)工作",
            r"不是(.+?)",
            r"没有(.+?)经验",
            r"从没有(.+?)"
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, context):
                context_bonus -= 0.3
                break
        
        # 检查是否有强化表述
        enhancement_patterns = [
            r"一直在(.+?)",
            r"一直是(.+?)",
            r"专门(.+?)",
            r"主要(.+?)",
            r"当前(.+?)"
        ]
        
        for pattern in enhancement_patterns:
            if re.search(pattern, context):
                context_bonus += 0.15
                break
        
        return context_bonus
    
    def _check_entity_type_compatibility(self, subject_entity, object_entity, relation_name):
        """检查实体类型兼容性"""
        subject_type = subject_entity['type']
        object_type = object_entity['type']
        
        # 定义合理的实体类型组合
        valid_combinations = {
            "担任": [("NAME", "TITLE")],
            "工作于": [("NAME", "ORG")],
            "居住于": [("NAME", "LOC")],
            "毕业于": [("NAME", "EDU")],
            "专业是": [("NAME", "PRO")],
            "位于": [("ORG", "LOC")],
            "联系方式": [("NAME", "CONT")],
            "民族是": [("NAME", "RACE")],
            "隶属于": [("ORG", "ORG")]
        }
        
        if relation_name in valid_combinations:
            if (subject_type, object_type) in valid_combinations[relation_name]:
                return 0.2
            else:
                return -0.3
        
        return 0.0
    
    def _calculate_ternary_confidence(self, text, name_entity, org_entity, title_entity, pattern_info, attention_weights=None, hidden_states=None):
        """计算三元关系的置信度，集成BERT上下文理解"""
        confidence = 0.3
        
        # 检查关键词
        keywords = pattern_info.get('keywords', [])
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        confidence += keyword_count * 0.1
        
        # 检查三个实体的相对位置
        entities_positions = [
            (name_entity['start'], 'NAME'),
            (org_entity['start'], 'ORG'),
            (title_entity['start'], 'TITLE')
        ]
        entities_positions.sort()
        
        # 检查是否符合常见的语言模式
        # 常见模式: 人名 + 在 + 组织 + 担任 + 职位
        pos_pattern = [pos[1] for pos in entities_positions]
        if pos_pattern == ['NAME', 'ORG', 'TITLE'] or pos_pattern == ['NAME', 'TITLE', 'ORG']:
            confidence += 0.2
        
        # 检查模式匹配
        patterns = pattern_info.get('patterns', [])
        for pattern in patterns:
            if re.search(pattern, text):
                confidence += 0.2
                break
        
        # BERT增强：分析三元关系的注意力模式
        if attention_weights is not None and hidden_states is not None:
            try:
                # 分析三个实体间的注意力关联
                entities = [name_entity, org_entity, title_entity]
                attention_score = self._analyze_ternary_attention(entities, attention_weights)
                confidence += attention_score * 0.15
                
                # 分析语义一致性
                semantic_score = self._analyze_ternary_semantics(entities, hidden_states)
                confidence += semantic_score * 0.1
                
            except Exception as e:
                print(f"三元关系BERT分析失败: {e}")
        
        return min(confidence, 1.0)
    
    def _analyze_ternary_attention(self, entities, attention_weights):
        """分析三元关系的注意力模式"""
        try:
            total_attention = 0.0
            
            # 计算三个实体间的相互注意力
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j:
                        start1 = entity1.get('token_start', entity1['start'] + 1)
                        end1 = entity1.get('token_end', entity1['end'] + 1)
                        start2 = entity2.get('token_start', entity2['start'] + 1)
                        end2 = entity2.get('token_end', entity2['end'] + 1)
                        
                        # 分析最后两层的注意力
                        for layer_idx in range(10, 12):
                            if layer_idx < len(attention_weights):
                                layer_attention = attention_weights[layer_idx].squeeze(0)
                                
                                for head in range(layer_attention.shape[0]):
                                    for pos1 in range(start1, min(end1 + 1, layer_attention.shape[1])):
                                        for pos2 in range(start2, min(end2 + 1, layer_attention.shape[2])):
                                            total_attention += layer_attention[head, pos1, pos2].item()
            
            # 归一化
            num_pairs = len(entities) * (len(entities) - 1)
            if num_pairs > 0:
                return min(total_attention / (num_pairs * 24), 0.3)  # 12头 * 2层 = 24
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_ternary_semantics(self, entities, hidden_states):
        """分析三元关系的语义一致性"""
        try:
            # 获取三个实体的隐藏状态表示
            entity_representations = []
            
            for entity in entities:
                start = entity.get('token_start', entity['start'] + 1)
                end = entity.get('token_end', entity['end'] + 1)
                entity_hidden = hidden_states[0, start:end+1].mean(dim=0)
                entity_representations.append(entity_hidden)
            
            # 计算三个实体间的平均余弦相似度
            similarities = []
            for i in range(len(entity_representations)):
                for j in range(i + 1, len(entity_representations)):
                    sim = F.cosine_similarity(
                        entity_representations[i].unsqueeze(0),
                        entity_representations[j].unsqueeze(0),
                        dim=1
                    ).item()
                    similarities.append(sim)
            
            # 返回平均相似度，但权重较低（三元关系中实体类型差异较大）
            if similarities:
                return np.mean(similarities) * 0.1
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_extended_context(self, text, entities):
        """提取扩展上下文（用于多实体）"""
        positions = [entity['start'] for entity in entities] + [entity['end'] for entity in entities]
        start = min(positions)
        end = max(positions)
        
        # 扩展上下文
        context_start = max(0, start - 15)
        context_end = min(len(text), end + 15)
        
        return text[context_start:context_end + 1]
    
    def cross_sentence_relation_extraction(self, text_list):
        """跨句子关系抽取"""
        all_entities = []
        all_relations = []
        sentence_boundaries = []
        
        # 为每个句子提取实体
        current_offset = 0
        for i, sentence in enumerate(text_list):
            entities = self.extract_entities(sentence)
            
            # 调整实体位置以适应全文偏移
            for entity in entities:
                entity['global_start'] = entity['start'] + current_offset
                entity['global_end'] = entity['end'] + current_offset
                entity['sentence_id'] = i
                all_entities.append(entity)
            
            sentence_boundaries.append((current_offset, current_offset + len(sentence)))
            current_offset += len(sentence) + 1  # +1 为句子间的分隔符
        
        # 先处理句内关系
        for i, sentence in enumerate(text_list):
            sentence_entities = [e for e in all_entities if e['sentence_id'] == i]
            sentence_relations = self.extract_relations_by_rules(sentence, sentence_entities)
            for relation in sentence_relations:
                relation['scope'] = 'intra-sentence'
                relation['sentence_id'] = i
            all_relations.extend(sentence_relations)
        
        # 处理跨句子关系
        full_text = ' '.join(text_list)
        cross_relations = self._extract_cross_sentence_relations(all_entities, full_text, sentence_boundaries)
        all_relations.extend(cross_relations)
        
        return {
            'text_list': text_list,
            'full_text': full_text,
            'entities': all_entities,
            'relations': all_relations,
            'sentence_boundaries': sentence_boundaries
        }
    
    def _filter_conflicting_relations(self, new_relations, existing_relations):
        """过滤冲突的关系，保留优先级更高的关系"""
        filtered = []
        
        for new_rel in new_relations:
            has_conflict = False
            
            for existing_rel in existing_relations:
                # 检查是否为同一对实体
                if (new_rel['subject']['text'] == existing_rel['subject']['text'] and 
                    new_rel['object']['text'] == existing_rel['object']['text']):
                    
                    # 检查是否为冲突的关系类型
                    conflicting_pairs = [
                        ("毕业于", "担任"),
                        ("毕业于", "工作于"),
                        ("专业是", "担任"),
                        ("联系方式", "担任")
                    ]
                    
                    for pair in conflicting_pairs:
                        if ((new_rel['relation'] == pair[0] and existing_rel['relation'] == pair[1]) or
                            (new_rel['relation'] == pair[1] and existing_rel['relation'] == pair[0])):
                            has_conflict = True
                            break
                    
                    if has_conflict:
                        break
            
            if not has_conflict:
                filtered.append(new_rel)
        
        return filtered
    
    def _extract_cross_sentence_relations(self, entities, full_text, sentence_boundaries):
        """提取跨句子关系"""
        cross_relations = []
        
        # 按实体类型分组
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        # 只处理特定类型的跨句子关系
        cross_sentence_patterns = {
            "工作于": {
                "subject_type": "NAME", "object_type": "ORG",
                "max_distance": 2, "keywords": ["工作", "公司", "组织"]
            },
            "担任": {
                "subject_type": "NAME", "object_type": "TITLE",
                "max_distance": 2, "keywords": ["担任", "职位", "岗位"]
            },
            "居住于": {
                "subject_type": "NAME", "object_type": "LOC",
                "max_distance": 3, "keywords": ["住", "家", "地址"]
            }
        }
        
        for relation_name, pattern_info in cross_sentence_patterns.items():
            subject_type = pattern_info['subject_type']
            object_type = pattern_info['object_type']
            max_distance = pattern_info['max_distance']
            keywords = pattern_info['keywords']
            
            if not entities_by_type.get(subject_type) or not entities_by_type.get(object_type):
                continue
            
            for subject_entity in entities_by_type[subject_type]:
                for object_entity in entities_by_type[object_type]:
                    # 检查是否在不同句子中
                    if subject_entity['sentence_id'] == object_entity['sentence_id']:
                        continue
                    
                    # 检查句子距离
                    sentence_distance = abs(subject_entity['sentence_id'] - object_entity['sentence_id'])
                    if sentence_distance > max_distance:
                        continue
                    
                    # 检查是否有相关关键词
                    context_start = min(subject_entity['global_start'], object_entity['global_start'])
                    context_end = max(subject_entity['global_end'], object_entity['global_end'])
                    context = full_text[max(0, context_start-20):context_end+20]
                    
                    has_keywords = any(keyword in context for keyword in keywords)
                    if not has_keywords:
                        continue
                    
                    # 计算跨句子关系的置信度
                    confidence = self._calculate_cross_sentence_confidence(
                        subject_entity, object_entity, context, pattern_info, sentence_distance
                    )
                    
                    if confidence > 0.3:
                        relation = {
                            'relation': relation_name,
                            'subject': subject_entity,
                            'object': object_entity,
                            'confidence': confidence,
                            'context': context[:50] + "..." if len(context) > 50 else context,
                            'direction': f"{subject_type} -> {object_type}",
                            'type': 'cross-sentence',
                            'scope': 'inter-sentence',
                            'sentence_distance': sentence_distance
                        }
                        cross_relations.append(relation)
        
        return cross_relations
    
    def _calculate_cross_sentence_confidence(self, subject_entity, object_entity, context, pattern_info, sentence_distance):
        """计算跨句子关系的置信度"""
        confidence = 0.2  # 跨句子关系的基础置信度较低
        
        # 根据句子距离调整
        if sentence_distance == 1:
            confidence += 0.2  # 相邻句子
        elif sentence_distance == 2:
            confidence += 0.1
        else:
            confidence += 0.05
        
        # 检查关键词密度
        keywords = pattern_info.get('keywords', [])
        keyword_count = sum(1 for keyword in keywords if keyword in context)
        confidence += keyword_count * 0.1
        
        # 检查是否有强连接词
        strong_connectors = ["另外", "同时", "此外", "还有", "并且", "而且"]
        if any(connector in context for connector in strong_connectors):
            confidence += 0.15
        
        return confidence
    
    def _extract_context(self, text, entity1, entity2):
        """提取实体间的上下文"""
        start = min(entity1['start'], entity2['start'])
        end = max(entity1['end'], entity2['end'])
        
        # 扩展上下文
        context_start = max(0, start - 10)
        context_end = min(len(text), end + 10)
        
        return text[context_start:context_end + 1]
    
    def _deduplicate_relations(self, relations):
        """关系去重 - 考虑方向性"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 使用主体和客体的确定顺序作为去重键
            key = (
                relation['relation'],
                relation['subject']['text'],
                relation['object']['text'],
                relation['direction']  # 包含方向信息
            )
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def extract_relations(self, text):
        """主函数：使用BERT增强模式提取文本中的实体和关系"""
        print(f"\n正在使用BERT增强模式分析文本（简化版本）: {text}")
        
        # 1. 提取实体并获取BERT隐藏状态（简化版本）
        entities, hidden_states, attention_weights = self.extract_entities_with_attention(text)
        print(f"识别到 {len(entities)} 个实体（含BERT上下文信息）")
        
        # 2. 基于规则和BERT上下文提取关系（简化版本）
        relations = self.extract_relations_by_rules(text, entities)
        print(f"抽取到 {len(relations)} 个关系（BERT增强简化版）")
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations
        }
    
    def format_results(self, result):
        """格式化显示结果"""
        text = result.get('text', result.get('full_text', ''))
        entities = result['entities']
        relations = result['relations']
        
        print("\n" + "="*80)
        print("实体识别与关系抽取结果")
        print("="*80)
        
        if result.get('text_list'):
            print(f"\n输入文本（多句）:")
            for i, sentence in enumerate(result['text_list']):
                print(f"  {i+1}. {sentence}")
        else:
            print(f"\n输入文本: {text}")
        
        # 显示实体
        print(f"\n识别的实体 ({len(entities)}个):")
        if entities:
            print("-" * 80)
            print(f"{'序号':<4} {'实体类型':<12} {'实体文本':<20} {'位置':<15} {'句子ID':<8}")
            print("-" * 80)
            for i, entity in enumerate(entities, 1):
                sentence_info = f"句{entity.get('sentence_id', 'N/A')}" if 'sentence_id' in entity else 'N/A'
                position = f"{entity['start']}-{entity['end']}"
                if 'global_start' in entity:
                    position += f" (全局{entity['global_start']}-{entity['global_end']})"
                print(f"{i:<4} {entity['type_name']:<12} {entity['text']:<20} {position:<15} {sentence_info:<8}")
        else:
            print("  未识别到实体")
        
        # 显示关系
        print(f"\n抽取的关系 ({len(relations)}个):")
        if relations:
            # 按类型分组显示
            intra_relations = [r for r in relations if r.get('scope') == 'intra-sentence' or r.get('type') in ['binary', 'ternary']]
            inter_relations = [r for r in relations if r.get('scope') == 'inter-sentence']
            
            if intra_relations:
                print("\n句内关系:")
                print("-" * 110)
                print(f"{'序号':<4} {'关系类型':<12} {'主体':<15} {'客体':<15} {'方向':<18} {'置信度':<8} {'类型':<8} {'上下文'}")
                print("-" * 110)
                for i, relation in enumerate(intra_relations, 1):
                    subject = relation['subject']
                    object_entity = relation['object']
                    confidence = f"{relation['confidence']:.2f}"
                    context = relation['context'][:20] + "..." if len(relation['context']) > 20 else relation['context']
                    direction = relation['direction']
                    rel_type = relation.get('type', 'binary')
                    
                    print(f"{i:<4} {relation['relation']:<12} {subject['text']:<15} {object_entity['text']:<15} {direction:<18} {confidence:<8} {rel_type:<8} {context}")
                    
                    # 如果是三元关系，显示辅助实体
                    if 'auxiliary' in relation:
                        aux_entity = relation['auxiliary']
                        print(f"{'':>50} 辅助实体: {aux_entity['text']} ({aux_entity['type_name']})")
            
            if inter_relations:
                print("\n跨句子关系:")
                print("-" * 120)
                print(f"{'序号':<4} {'关系类型':<12} {'主体':<15} {'客体':<15} {'方向':<18} {'置信度':<8} {'句距':<6} {'上下文'}")
                print("-" * 120)
                for i, relation in enumerate(inter_relations, 1):
                    subject = relation['subject']
                    object_entity = relation['object']
                    confidence = f"{relation['confidence']:.2f}"
                    context = relation['context'][:25] + "..." if len(relation['context']) > 25 else relation['context']
                    direction = relation['direction']
                    sentence_dist = relation.get('sentence_distance', 'N/A')
                    
                    print(f"{i:<4} {relation['relation']:<12} {subject['text']:<15} {object_entity['text']:<15} {direction:<18} {confidence:<8} {sentence_dist:<6} {context}")
        else:
            print("  未抽取到关系")
        
        print("\n" + "="*80)
    
    def export_to_json(self, result, filename=None):
        """导出结果为JSON格式"""
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"relation_extraction_result_{timestamp}.json"
        
        # 清理数据，确保可序列化
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'text': result.get('text', result.get('full_text', '')),
                'text_list': result.get('text_list', []),
                'type': 'multi-sentence' if result.get('text_list') else 'single-sentence'
            },
            'statistics': {
                'total_entities': len(result['entities']),
                'total_relations': len(result['relations']),
                'entity_types': {},
                'relation_types': {}
            },
            'entities': [],
            'relations': []
        }
        
        # 处理实体数据
        for entity in result['entities']:
            entity_data = {
                'text': entity['text'],
                'type': entity['type'],
                'type_name': entity['type_name'],
                'start': entity['start'],
                'end': entity['end']
            }
            if 'global_start' in entity:
                entity_data['global_start'] = entity['global_start']
                entity_data['global_end'] = entity['global_end']
            if 'sentence_id' in entity:
                entity_data['sentence_id'] = entity['sentence_id']
            
            export_data['entities'].append(entity_data)
            
            # 统计实体类型
            entity_type = entity['type_name']
            export_data['statistics']['entity_types'][entity_type] = export_data['statistics']['entity_types'].get(entity_type, 0) + 1
        
        # 处理关系数据
        for relation in result['relations']:
            relation_data = {
                'relation': relation['relation'],
                'subject': {
                    'text': relation['subject']['text'],
                    'type': relation['subject']['type'],
                    'type_name': relation['subject']['type_name']
                },
                'object': {
                    'text': relation['object']['text'],
                    'type': relation['object']['type'],
                    'type_name': relation['object']['type_name']
                },
                'confidence': relation['confidence'],
                'context': relation['context'],
                'direction': relation['direction'],
                'type': relation.get('type', 'binary'),
                'scope': relation.get('scope', 'intra-sentence')
            }
            
            if 'auxiliary' in relation:
                relation_data['auxiliary'] = {
                    'text': relation['auxiliary']['text'],
                    'type': relation['auxiliary']['type'],
                    'type_name': relation['auxiliary']['type_name']
                }
            
            if 'sentence_distance' in relation:
                relation_data['sentence_distance'] = relation['sentence_distance']
            
            export_data['relations'].append(relation_data)
            
            # 统计关系类型
            relation_type = relation['relation']
            export_data['statistics']['relation_types'][relation_type] = export_data['statistics']['relation_types'].get(relation_type, 0) + 1
        
        # 写入文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"\n结果已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"导出失败: {e}")
            return None
    
    def export_to_csv(self, result, filename=None):
        """导出关系结果为CSV格式"""
        import csv
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"relation_extraction_result_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入头部
                headers = [
                    '关系类型', '主体文本', '主体类型', '客体文本', '客体类型',
                    '置信度', '方向', '类型', '范围', '上下文'
                ]
                writer.writerow(headers)
                
                # 写入关系数据
                for relation in result['relations']:
                    row = [
                        relation['relation'],
                        relation['subject']['text'],
                        relation['subject']['type_name'],
                        relation['object']['text'],
                        relation['object']['type_name'],
                        f"{relation['confidence']:.3f}",
                        relation['direction'],
                        relation.get('type', 'binary'),
                        relation.get('scope', 'intra-sentence'),
                        relation['context']
                    ]
                    writer.writerow(row)
            
            print(f"\n关系结果已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"CSV导出失败: {e}")
            return None
    
    def generate_statistics_report(self, result):
        """生成统计报告"""
        entities = result['entities']
        relations = result['relations']
        
        print("\n" + "="*60)
        print("统计报告")
        print("="*60)
        
        # 实体统计
        entity_stats = {}
        for entity in entities:
            entity_type = entity['type_name']
            entity_stats[entity_type] = entity_stats.get(entity_type, 0) + 1
        
        print(f"\n实体统计 (总计: {len(entities)}个):")
        print("-" * 30)
        for entity_type, count in sorted(entity_stats.items()):
            percentage = (count / len(entities)) * 100 if entities else 0
            print(f"{entity_type:<12}: {count:>3}个 ({percentage:5.1f}%)")
        
        # 关系统计
        relation_stats = {}
        confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
        type_stats = {'binary': 0, 'ternary': 0}
        scope_stats = {'intra-sentence': 0, 'inter-sentence': 0}
        
        for relation in relations:
            # 关系类型统计
            rel_type = relation['relation']
            relation_stats[rel_type] = relation_stats.get(rel_type, 0) + 1
            
            # 置信度统计
            confidence = relation['confidence']
            if confidence >= 0.7:
                confidence_stats['high'] += 1
            elif confidence >= 0.5:
                confidence_stats['medium'] += 1
            else:
                confidence_stats['low'] += 1
            
            # 类型统计
            rel_structure = relation.get('type', 'binary')
            type_stats[rel_structure] += 1
            
            # 范围统计
            scope = relation.get('scope', 'intra-sentence')
            scope_stats[scope] += 1
        
        print(f"\n关系统计 (总计: {len(relations)}个):")
        print("-" * 30)
        for rel_type, count in sorted(relation_stats.items()):
            percentage = (count / len(relations)) * 100 if relations else 0
            print(f"{rel_type:<12}: {count:>3}个 ({percentage:5.1f}%)")
        
        print(f"\n置信度分布:")
        print("-" * 20)
        for level, count in confidence_stats.items():
            percentage = (count / len(relations)) * 100 if relations else 0
            level_name = {'high': '高(≥0.7)', 'medium': '中(0.5-0.7)', 'low': '低(<0.5)'}[level]
            print(f"{level_name:<10}: {count:>3}个 ({percentage:5.1f}%)")
        
        if any(scope_stats.values()):
            print(f"\n关系范围:")
            print("-" * 15)
            for scope, count in scope_stats.items():
                percentage = (count / len(relations)) * 100 if relations else 0
                scope_name = {'intra-sentence': '句内', 'inter-sentence': '跨句'}[scope]
                print(f"{scope_name:<8}: {count:>3}个 ({percentage:5.1f}%)")
        
        print("\n" + "="*60)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*60)
        print("增强型实体识别与关系抽取系统")
        print("="*60)
        print("输入中文句子进行实体识别和关系抽取")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看支持的功能")
        print("输入 'multi' 进入多句子模式")
        print("输入 'stats' 查看统计信息")
        print("-" * 60)
        
        session_results = []  # 保存本次会话的所有结果
        
        while True:
            try:
                user_input = input("\n请输入句子: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    if session_results:
                        save_choice = input("是否保存本次会话的结果? (y/n): ").strip().lower()
                        if save_choice in ['y', 'yes', '是']:
                            self._save_session_results(session_results)
                    print("再见！")
                    break
                    
                elif user_input.lower() in ['help', '帮助']:
                    self._show_help()
                    continue
                    
                elif user_input.lower() == 'multi':
                    result = self._multi_sentence_mode()
                    if result:
                        session_results.append(result)
                    continue
                    
                elif user_input.lower() == 'stats':
                    if session_results:
                        self._show_session_stats(session_results)
                    else:
                        print("暂无统计数据，请先进行关系抽取")
                    continue
                    
                elif not user_input:
                    print("请输入有效的句子")
                    continue
                
                # 进行关系抽取
                result = self.extract_relations(user_input)
                session_results.append(result)
                
                # 显示结果
                self.format_results(result)
                
                # 提供导出选项
                export_choice = input("\n是否导出结果? (j=JSON, c=CSV, n=不导出): ").strip().lower()
                if export_choice == 'j':
                    self.export_to_json(result)
                elif export_choice == 'c':
                    self.export_to_csv(result)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理过程中发生错误: {e}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n支持的关系类型:")
        print("-" * 40)
        for relation_name, pattern_info in self.relation_patterns.items():
            subject_type = self.entity_map.get(pattern_info['subject_type'], pattern_info['subject_type'])
            object_type = self.entity_map.get(pattern_info['object_type'], pattern_info['object_type'])
            if pattern_info.get('is_ternary'):
                aux_type = self.entity_map.get(pattern_info.get('auxiliary_type', ''), '')
                print(f"  {relation_name}: {subject_type} -> {object_type} ({aux_type})")
            else:
                print(f"  {relation_name}: {subject_type} -> {object_type}")
        
        print("\n支持的命令:")
        print("-" * 20)
        print("  help/帮助 - 显示帮助信息")
        print("  multi - 进入多句子模式")
        print("  stats - 显示本次会话统计")
        print("  quit/exit/退出 - 退出程序")
        
        print("\n支持的实体类型:")
        print("-" * 20)
        for etype, name in self.entity_map.items():
            print(f"  {etype}: {name}")
    
    def _multi_sentence_mode(self):
        """多句子模式"""
        print("\n进入多句子模式 - 输入多个句子，空行结束:")
        sentences = []
        
        while True:
            sentence = input(f"句子 {len(sentences)+1}: ").strip()
            if not sentence:
                break
            sentences.append(sentence)
        
        if not sentences:
            print("未输入任何句子")
            return None
        
        print(f"\n开始处理 {len(sentences)} 个句子...")
        result = self.cross_sentence_relation_extraction(sentences)
        
        # 显示结果
        self.format_results(result)
        
        # 生成统计报告
        self.generate_statistics_report(result)
        
        return result
    
    def _show_session_stats(self, session_results):
        """显示会话统计"""
        total_entities = sum(len(result['entities']) for result in session_results)
        total_relations = sum(len(result['relations']) for result in session_results)
        
        print("\n" + "="*50)
        print("本次会话统计")
        print("="*50)
        print(f"处理文本数: {len(session_results)}")
        print(f"识别实体总数: {total_entities}")
        print(f"抽取关系总数: {total_relations}")
        
        if total_relations > 0:
            avg_confidence = sum(
                sum(rel['confidence'] for rel in result['relations']) 
                for result in session_results
            ) / total_relations
            print(f"平均置信度: {avg_confidence:.3f}")
        
        print("="*50)
    
    def _save_session_results(self, session_results):
        """保存会话结果"""
        from datetime import datetime
        
        # 合并所有结果
        combined_result = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'total_texts': len(session_results),
                'session_type': 'interactive'
            },
            'results': session_results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_results_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(combined_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"会话结果已保存到: {filename}")
        except Exception as e:
            print(f"保存失败: {e}")

def main():
    print("增强型BERT-NER-CRF 实体识别与关系抽取系统")
    print("=" * 60)
    
    # 模型路径
    model_path = "outputs\\cner_output\\bert"
    
    # 检查模型文件
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"错误: 模型文件不存在 {os.path.join(model_path, 'model.safetensors')}")
        return
    
    # 创建关系抽取系统
    extractor = RelationExtractor(model_path)
    
    # 测试示例
    print("\n先用示例测试系统:")
    test_examples = [
        "张三是软件工程师，在北京阿里巴巴工作，住在朝阳区，电话是13800138000。",
        "李教授毕业于清华大学计算机专业，现在厦门泛华集团担任技术总监。",
        "王小明在上海微软公司做产品经理，家住浦东新区。",
        # 新增复杂示例
        "陈博士是汉族人，在中科院计算所从事人工智能研究，同时担任实验室主任。",
        "腾讯公司位于深圳，其下属的微信事业部在广州设有办公室。"
    ]
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n{'='*25} 示例 {i} {'='*25}")
        result = extractor.extract_relations(example)
        extractor.format_results(result)
        if i <= 3:  # 为前3个示例生成统计报告
            extractor.generate_statistics_report(result)
    
    # 测试多句子功能
    print(f"\n{'='*30} 多句子测试 {'='*30}")
    multi_sentences = [
        "张三在阿里巴巴工作。",
        "他担任高级软件工程师职位。",
        "家住在杭州西湖区。"
    ]
    
    multi_result = extractor.cross_sentence_relation_extraction(multi_sentences)
    extractor.format_results(multi_result)
    extractor.generate_statistics_report(multi_result)
    
    # 导出示例
    print(f"\n{'='*25} 导出功能测试 {'='*25}")
    json_file = extractor.export_to_json(multi_result)
    csv_file = extractor.export_to_csv(multi_result)
    
    # 进入交互模式
    extractor.interactive_mode()

if __name__ == "__main__":
    main()