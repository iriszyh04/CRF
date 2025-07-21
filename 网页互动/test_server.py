#!/usr/bin/env python3
"""
简化测试服务器
用于验证Python和Flask环境是否正常工作
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flask测试服务器</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .success { color: green; }
            .info { color: blue; }
        </style>
    </head>
    <body>
        <h1 class="success">✓ Flask服务器正常运行！</h1>
        <p class="info">如果您看到这个页面，说明Python和Flask环境正常。</p>
        
        <h2>测试链接：</h2>
        <ul>
            <li><a href="/test">测试API接口</a></li>
            <li><a href="/status">服务状态</a></li>
        </ul>
        
        <p><strong>下一步：</strong> 返回运行完整的app.py</p>
    </body>
    </html>
    '''

@app.route('/test')
def test():
    return jsonify({
        "status": "success",
        "message": "API接口正常工作",
        "python_version": "已连接"
    })

@app.route('/status')
def status():
    return jsonify({
        "server": "运行中",
        "flask": "正常",
        "port": 5000
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Flask测试服务器启动")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("如果可以访问，说明环境正常")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")