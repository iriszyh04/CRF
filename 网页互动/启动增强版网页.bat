@echo off
echo ===============================================
echo 增强型BERT-NER-CRF 网页界面启动脚本
echo ===============================================
echo.
echo 功能包括:
echo - 实体识别 (NER)
echo - 关系抽取 (Relation Extraction)
echo - 模型性能可视化 (Performance Visualization)
echo.
echo ===============================================

cd /d "C:\Users\Administrator\BERT-NER-Pytorch-master\BERT-NER-Pytorch-master\网页互动"

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo Python未安装或未添加到PATH
    echo 请下载安装Python: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 正在检查Flask是否已安装...
python -c "import flask; print('Flask版本:', flask.__version__)" 2>nul
if errorlevel 1 (
    echo Flask未安装，正在安装...
    pip install flask
    if errorlevel 1 (
        echo Flask安装失败，请手动运行: pip install flask
        pause
        exit /b 1
    )
) else (
    echo Flask已安装
)

echo.
echo 正在检查项目文件...
if not exist "app.py" (
    echo 错误: app.py 文件不存在
    pause
    exit /b 1
)

if not exist "model_analyzer.py" (
    echo 错误: model_analyzer.py 文件不存在
    pause
    exit /b 1
)

if not exist "templates\visualization.html" (
    echo 错误: templates\visualization.html 文件不存在
    pause
    exit /b 1
)

echo 项目文件检查完成

echo.
echo 正在检查模型文件...
if not exist "..\outputs\cner_output\bert\model.safetensors" (
    echo 警告: 模型文件可能不存在
    echo 路径: ..\outputs\cner_output\bert\model.safetensors
)

if not exist "..\outputs\cner_output\bert\test_performance_report.json" (
    echo 警告: 性能报告文件可能不存在
    echo 路径: ..\outputs\cner_output\bert\test_performance_report.json
)

echo.
echo 启动增强型Web服务...
echo ===============================================
echo 启动后可访问:
echo - 首页: http://localhost:5000
echo - 实体识别: http://localhost:5000/entity
echo - 关系抽取: http://localhost:5000/relation
echo - 性能可视化: http://localhost:5000/visualization
echo ===============================================
echo.

python app.py

echo.
echo 服务已停止
pause