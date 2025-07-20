# CRF 命名实体识别（NER）工具

基于条件随机场（CRF）模型的命名实体识别项目,可从文本中自动识别人名（PER）、地名（LOC）、组织机构名（ORG）、日期（DATE）等实体。

**示例**：​
- 输入文本：张三于2023年3月加入阿里巴巴，工作地点在杭州。​
- 输出实体：张三（PER）、2023年3月（DATE）、阿里巴巴（ORG）、杭州（LOC）

项目核心价值在于通过传统序列标注模型的特征工程实践，为自然语言处理基础任务提供可复用的解决方案，同时支持与深度学习模型（如Albert）结合的扩展方向。

## 项目目标  
- 实现基于 CRF 的序列标注能力，解决 NER 任务中的实体边界定位问题。  
- 探索特征工程对模型性能的影响（如词性、上下文窗口特征），为传统模型在 NER 中的应用提供参考。  
- 提供轻量化训练/预测工具，支持小批量文本的实体识别。
  
## 环境准备 
1. 克隆仓库到本地：​
```bash​
git clone https://github.com/iriszyh04/CRF.git​
cd CRF
```

2.安装依赖库：
```bash​
pip install -r requirements.txt
```

## 数据准备 
### 数据格式&#xA;

训练 / 测试数据需采用**BIO 标注体系**，具体规则：
*   每行格式：`字符 标签`（空格分隔）
*   句子间用空行分隔
*   标签定义：
    *   `B-X`：实体 X 的开始（如 B-PER 表示人名开始）
    *   `I-X`：实体 X 的中间 / 结尾（如 I-ORG 表示组织机构中间部分）
    *   `O`：非实体

**示例数据**（`data/train.txt`）：
```
张 B-PER
三 I-PER
在 O
2 B-DATE
0 I-DATE
2 I-DATE
3 I-DATE
年 I-DATE
加 O
入 O
百 B-ORG
度 I-ORG
北 B-LOC
京 I-LOC
是 O
中 B-LOC
国 I-LOC
的 O
首 B-LOC
都 I-LOC
```

## 使用指南
### 模型训练&#xA;

运行训练脚本，支持自定义参数配置：
```
python src/train.py \\
&#x20; \--train\_path ./data/train.txt \\
&#x20; \--model\_save\_path ./models/crf\_model.pkl \\
&#x20; \--max\_iter 160 \\
```

**参数说明**：
*   `--train_path`：训练数据路径（必填）
*   `--model_save_path`：模型保存路径（默认`./models/crf_model.pkl`）
*   `--max_iter`：训练迭代次数（默认 160）
  
训练过程中会输出每轮迭代的准确率、召回率和验证集 F1 分数，最终模型保存至指定路径。

### 实体预测&#xA;
支持单句预测和批量预测两种模式：
#### 1. 单句预测&#xA;
直接输入文本进行实体识别：
```
python src/predict.py \\
&#x20; \--text "李四在2024年访问上海交通大学" \\
&#x20; \--model\_path ./models/crf\_model.pkl
```
**输出示例**：
```
实体识别结果：
\- 李四：PER（位置：0-2）
\- 2024年：DATE（位置：3-8）
\- 上海交通大学：ORG（位置：10-16）
```

#### 2. 批量预测&#xA;
对测试集文件进行批量预测并保存结果：
```
python src/predict.py \\
&#x20; \--test\_path ./data/test.txt \\
&#x20; \--model\_path ./models/crf\_model.pkl \\
&#x20; \--output\_path ./results/predictions.txt
```

预测结果文件格式：每行对应输入句子的实体列表，格式为`实体文本:标签（起始位置-结束位置）`。

## 项目结构
```
项目仓库名/
├── mid_data/                
│   ├── cner_150_cut.txt           
│   ├── dev.json                  
│   ├── labels.json                
│   ├── nor_ent2id.json           
│   ├── test.json                  
│   ├── train.json               
│   └── train_aug.json            
├── albert_model/                  
│   ├── config.json                
│   ├── gitattributes              
│   ├── special_tokens_map.json    
│   ├── vocab.txt                    
├── model/                         
│   ├── logs/                      
│   │   ├── albert_base_model.py   
│   │   ├── albertcrf_model.py     
│   │   ├── config.py              
│   │   ├── crf.py                 
│   │   ├── cut.py                 
│   │   ├── dataset.py             
│   │   ├── main.py                
│   │   ├── predict.py             
│   │   ├── predict_grcq.py        
│   │   ├── preprocess.py          
│   │   └── utils                   
├── final data/                    
│   ├── dev.pkl                    
│   ├──test .pkl                   
│   ├──train.pkl                   
├── utils/                          
│   ├── __init__.py               
│   ├── commonUtils.py             
│   ├── cutSentences.py            
│   ├── decodeUtils.py             
│   ├── metricsUtils.py            
│   └── trainUtils.py               
└── README.md                      
```

## 免责声明
本项目为学术研究用途，模型性能受训练数据质量影响，不保证在生产环境中的稳定性。如需商业应用，建议进行充分的测试与优化。
