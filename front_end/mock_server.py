from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Create Flask app; static files are served from 'static' folder by default
app = Flask(__name__)
CORS(app)

# Fallback mapping from emotion to style
emotion_to_style = {
    'joy': 'Impressionism',
    'anger': 'Expressionism',
    'surprise': 'Surrealism',
    'disgust': 'Abstract',
    'fear': 'Gothic',
    'sadness': 'Baroque'
}

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    # Simulate processing delay
    time.sleep(1)
    # Dummy emotion scores
    emotions = {
        'anger': 0.05,
        'disgust': 0.02,
        'fear': 0.12,
        'joy': 0.65,
        'sadness': 0.08,
        'surprise': 0.08
    }
    # Determine dominant emotion
    dominant = max(emotions, key=emotions.get)
    # Respond with mock analysis
    return jsonify({
        'success': True,
        'emotionAnalysis': emotions,
        'dominantEmotion': dominant,
        'matchedStyle': emotion_to_style.get(dominant)
    })

@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    # Simulate processing delay
    time.sleep(1.5)
    # Return URL to a static dummy image
    return jsonify({
        'success': True,
        'styledImageUrl': '/static/styled.jpg'
    })

if __name__ == '__main__':
    # Run on port 5000 to match frontend expectations
    app.run(host='0.0.0.0', port=5000, debug=True)
