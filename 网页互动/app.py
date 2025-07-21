#!/usr/bin/env python3
"""
Flask网页后端服务
集成NER实体识别和关系抽取的Web界面
支持Windows和WSL路径
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import traceback
import platform

# 检测操作系统并设置正确的路径
def get_project_path():
    """根据操作系统获取正确的项目路径"""
    if platform.system() == "Windows":
        return r"C:\Users\Administrator\BERT-NER-Pytorch-master\BERT-NER-Pytorch-master"
    else:
        return "/mnt/c/Users/Administrator/BERT-NER-Pytorch-master/BERT-NER-Pytorch-master"

# 添加父目录到Python路径，以便导入我们的模块
project_path = get_project_path()
sys.path.append(project_path)
print(f"项目路径: {project_path}")

# 全局变量存储模型实例
ner_model = None
relation_model = None

def safe_import_models():
    """安全导入模型类"""
    try:
        from interactive_ner import InteractiveNER
        from relation_extraction import RelationExtractor
        print("✓ 成功导入NER和关系抽取模块")
        return InteractiveNER, RelationExtractor
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        # 创建占位符类
        class DummyNER:
            def __init__(self, model_path):
                self.model_path = model_path
            def predict_text(self, text):
                return {"error": "NER模块导入失败", "text": text, "entities": []}
        
        class DummyRelation:
            def __init__(self, model_path):
                self.model_path = model_path
            def extract_relations(self, text):
                return {"error": "关系抽取模块导入失败", "text": text, "entities": [], "relations": []}
        
        return DummyNER, DummyRelation

# 导入模型类
InteractiveNER, RelationExtractor = safe_import_models()

# 导入模型分析器
def safe_import_analyzer():
    """安全导入模型分析器"""
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        from model_analyzer import ModelAnalyzer
        print("✓ 成功导入模型分析器")
        return ModelAnalyzer
    except ImportError as e:
        print(f"✗ 导入模型分析器失败: {e}")
        # 创建占位符类
        class DummyAnalyzer:
            def __init__(self, model_output_dir=None):
                self.model_output_dir = model_output_dir
            def load_all_data(self):
                return False
            def get_all_visualization_data(self):
                return {"error": "模型分析器导入失败"}
        return DummyAnalyzer

ModelAnalyzer = safe_import_analyzer()

app = Flask(__name__)

# 全局变量存储分析器实例
model_analyzer = None

def initialize_models():
    """初始化模型"""
    global ner_model, relation_model, model_analyzer
    
    # 根据操作系统设置模型路径
    if platform.system() == "Windows":
        model_path = r"C:\Users\Administrator\BERT-NER-Pytorch-master\BERT-NER-Pytorch-master\outputs\cner_output\bert"
    else:
        model_path = "/mnt/c/Users/Administrator/BERT-NER-Pytorch-master/BERT-NER-Pytorch-master/outputs/cner_output/bert"
    
    print(f"模型路径: {model_path}")
    
    try:
        print("正在初始化NER模型...")
        ner_model = InteractiveNER(model_path)
        print("✓ NER模型初始化成功")
    except Exception as e:
        print(f"✗ NER模型初始化失败: {e}")
        ner_model = None
    
    try:
        print("正在初始化关系抽取模型...")
        relation_model = RelationExtractor(model_path)
        print("✓ 关系抽取模型初始化成功")
    except Exception as e:
        print(f"✗ 关系抽取模型初始化失败: {e}")
        relation_model = None
    
    try:
        print("正在初始化模型分析器...")
        model_analyzer = ModelAnalyzer(model_path)
        if model_analyzer.load_all_data():
            print("✓ 模型分析器初始化成功")
        else:
            print("✗ 模型分析器数据加载失败")
    except Exception as e:
        print(f"✗ 模型分析器初始化失败: {e}")
        model_analyzer = None

# 确保静态文件目录存在
os.makedirs(os.path.join(app.root_path, 'static', 'js'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 实体识别页面
@app.route('/entity')
def entity_page():
    return render_template('entity.html')

# 关系抽取页面
@app.route('/relation')
def relation_page():
    return render_template('relation.html')

# 可视化页面（保持兼容）
@app.route('/visualization')
def visualization_page():
    return render_template('visualization.html')

# API路由
@app.route('/api/ner', methods=['POST'])
def api_ner():
    """NER实体识别API"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "请输入文本"})
        
        if not ner_model:
            return jsonify({"error": "NER模型未初始化"})
        
        # 调用NER模型
        result = ner_model.predict_text(text)
        
        if result and 'entities' in result:
            return jsonify({
                "success": True,
                "text": text,
                "entities": result['entities'],
                "entity_count": len(result['entities'])
            })
        else:
            return jsonify({"error": "NER预测失败"})
            
    except Exception as e:
        print(f"NER API错误: {e}")
        traceback.print_exc()
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/relation', methods=['POST'])
def api_relation():
    """关系抽取API"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "请输入文本"})
        
        if not relation_model:
            return jsonify({"error": "关系抽取模型未初始化"})
        
        # 调用关系抽取模型
        result = relation_model.extract_relations(text)
        
        if result and 'entities' in result and 'relations' in result:
            return jsonify({
                "success": True,
                "text": text,
                "entities": result['entities'],
                "relations": result['relations'],
                "entity_count": len(result['entities']),
                "relation_count": len(result['relations'])
            })
        else:
            return jsonify({"error": "关系抽取失败"})
            
    except Exception as e:
        print(f"关系抽取API错误: {e}")
        traceback.print_exc()
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/status')
def api_status():
    """系统状态API"""
    return jsonify({
        "ner_model": ner_model is not None,
        "relation_model": relation_model is not None,
        "model_analyzer": model_analyzer is not None,
        "status": "运行中"
    })

@app.route('/api/examples')
def api_examples():
    """示例数据API"""
    examples = [
        "我叫张三，住在北京市朝阳区，在清华大学工作。",
        "王小明是软件工程师，他的电话是13800138000。", 
        "李教授毕业于清华大学计算机专业，现在厦门泛华集团担任技术总监。",
        "刘博士在上海复旦大学任教，研究人工智能领域。",
        "陈经理来自广州，在腾讯公司做产品经理，联系邮箱是chen@qq.com。"
    ]
    return jsonify(examples)

@app.route('/api/visualization-data')
def api_visualization_data():
    """模型可视化数据API"""
    try:
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        # 获取可视化数据
        viz_data = model_analyzer.get_all_visualization_data()
        
        if 'error' in viz_data:
            return jsonify({"error": viz_data['error']})
        
        return jsonify({
            "success": True,
            **viz_data
        })
        
    except Exception as e:
        print(f"可视化数据API错误: {e}")
        traceback.print_exc()
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/model-info')
def api_model_info():
    """模型信息API"""
    try:
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        # 获取模型基本信息
        model_info = {
            "model_type": "BERT-CRF",
            "model_name": "bert-base-chinese",
            "task": "Chinese Named Entity Recognition",
            "dataset": "CNER",
            "entity_types": ["NAME", "ORG", "TITLE", "EDU", "LOC", "CONT", "PRO", "RACE"],
            "entity_type_names": {
                "NAME": "姓名",
                "ORG": "组织机构", 
                "TITLE": "职位",
                "EDU": "教育背景",
                "LOC": "地址",
                "CONT": "联系方式",
                "PRO": "专业",
                "RACE": "民族"
            }
        }
        
        return jsonify({
            "success": True,
            **model_info
        })
        
    except Exception as e:
        print(f"模型信息API错误: {e}")
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/training-progress')
def api_training_progress():
    """训练进度数据API"""
    try:
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        progress_data = model_analyzer.get_training_progress_data()
        
        if not progress_data:
            return jsonify({"error": "训练进度数据不可用"})
        
        return jsonify({
            "success": True,
            "training_progress": progress_data
        })
        
    except Exception as e:
        print(f"训练进度API错误: {e}")
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/entity-performance')
def api_entity_performance():
    """实体性能数据API"""
    try:
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        entity_summary = model_analyzer.get_entity_performance_summary()
        
        if not entity_summary:
            return jsonify({"error": "实体性能数据不可用"})
        
        return jsonify({
            "success": True,
            "entity_summary": entity_summary
        })
        
    except Exception as e:
        print(f"实体性能API错误: {e}")
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/refresh-data', methods=['POST'])
def api_refresh_data():
    """刷新模型数据API"""
    try:
        global model_analyzer
        
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        # 重新加载数据
        if model_analyzer.load_all_data():
            return jsonify({
                "success": True,
                "message": "数据刷新成功"
            })
        else:
            return jsonify({"error": "数据刷新失败"})
        
    except Exception as e:
        print(f"数据刷新API错误: {e}")
        return jsonify({"error": f"服务器错误: {str(e)}"})

@app.route('/api/export-report', methods=['POST'])
def api_export_report():
    """导出性能报告API"""
    try:
        if not model_analyzer:
            return jsonify({"error": "模型分析器未初始化"})
        
        # 获取报告数据
        viz_data = model_analyzer.get_all_visualization_data()
        
        # 构建报告内容
        report = {
            "report_time": viz_data.get('model_info', {}).get('last_updated', ''),
            "overall_metrics": viz_data.get('overall_metrics', {}),
            "entity_summary": viz_data.get('entity_summary', {}),
            "training_info": viz_data.get('training_progress', {}).get('training_params', {})
        }
        
        return jsonify({
            "success": True,
            "report": report
        })
        
    except Exception as e:
        print(f"导出报告API错误: {e}")
        return jsonify({"error": f"服务器错误: {str(e)}"})

if __name__ == '__main__':
    print("=== 增强型BERT-NER-CRF Web服务启动 ===")
    
    # 初始化模型
    initialize_models()
    
    # 启动Flask应用
    print("启动Web服务器...")
    print("访问地址: http://localhost:5000")
    print("实体识别: http://localhost:5000/entity")
    print("关系抽取: http://localhost:5000/relation")
    print("性能可视化: http://localhost:5000/visualization")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
