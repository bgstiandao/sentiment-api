import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 关键设置

from flask import Flask, request, jsonify
from transformers import pipeline
import time

app = Flask(__name__)
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0f"

# 添加模型加载状态
model_loaded = False
loading_start_time = 0

@app.route('/')
def home():
    return "情感分析API已就绪！请POST文本到/analyze"

@app.route('/analyze', methods=['POST'])
def analyze_text():
    global model_loaded, sentiment_analyzer, loading_start_time

    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "请提供文本内容"}), 400

    try:
        # 检查模型是否已加载
        if not model_loaded:
            loading_start_time = time.time()
            print("开始加载模型...")
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=MODEL_NAME,
                revision=MODEL_REVISION
            )
            model_loaded = True
            print(f"模型加载完成! 耗时: {time.time() - loading_start_time:.2f}秒")

        result = sentiment_analyzer(text)[0]
        return jsonify({
            "text": text,
            "sentiment": result['label'],
            "confidence": round(result['score'], 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({
        "status": "ready" if model_loaded else "loading",
        "model": MODEL_NAME,
        "elapsed": time.time() - loading_start_time if loading_start_time > 0 else 0
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)