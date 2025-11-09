from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import threading
import subprocess
import json
from pathlib import Path

app = Flask(__name__)

# Global state for training progress
training_state = {
    'model1': {'running': False, 'progress': 0, 'status': 'Not started', 'accuracy': None, 'error': None},
    'model2': {'running': False, 'progress': 0, 'status': 'Not started', 'accuracy': None, 'error': None}
}

def check_model_exists(model_name):
    """Check if model artifact exists"""
    if model_name == 'model1':
        return os.path.exists('model1.joblib')
    elif model_name == 'model2':
        return os.path.exists('model2.keras')
    return False

def check_tensorflow():
    """Check if TensorFlow is available"""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/env')
def env_info():
    """Return environment information"""
    import platform
    return jsonify({
        'python': sys.version,
        'arch': platform.machine(),
        'tf_available': check_tensorflow()
    })

@app.route('/model_status')
def model_status():
    """Return status of model artifacts"""
    return jsonify({
        'model1': check_model_exists('model1'),
        'model2': check_model_exists('model2'),
        'tf_available': check_tensorflow()
    })

@app.route('/train/<model_name>', methods=['POST'])
def train_model(model_name):
    """Start training a model in background"""
    if model_name not in ['model1', 'model2']:
        return jsonify({'error': 'Invalid model name'}), 400
    
    if training_state[model_name]['running']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Check TensorFlow for model2
    if model_name == 'model2' and not check_tensorflow():
        return jsonify({'error': 'TensorFlow not installed'}), 400
    
    # Reset state
    training_state[model_name] = {
        'running': True,
        'progress': 0,
        'status': 'Starting training...',
        'accuracy': None,
        'error': None
    }
    
    # Start training in background thread
    thread = threading.Thread(target=run_training, args=(model_name,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})

def run_training(model_name):
    """Run training script and monitor progress"""
    try:
        script_path = f'trainers/{model_name}_trainer.py'
        
        if not os.path.exists(script_path):
            training_state[model_name]['error'] = f'Trainer script not found: {script_path}'
            training_state[model_name]['running'] = False
            return
        
        # Run the training script
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor output
        for line in process.stdout:
            line = line.strip()
            if line.startswith('PROGRESS:'):
                try:
                    data = json.loads(line[9:])
                    training_state[model_name]['progress'] = data.get('progress', 0)
                    training_state[model_name]['status'] = data.get('status', '')
                    if 'accuracy' in data:
                        training_state[model_name]['accuracy'] = data['accuracy']
                except json.JSONDecodeError:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            training_state[model_name]['error'] = f'Training failed: {stderr}'
            training_state[model_name]['status'] = 'Training failed'
        else:
            training_state[model_name]['status'] = 'Training complete'
            training_state[model_name]['progress'] = 100
        
    except Exception as e:
        training_state[model_name]['error'] = str(e)
        training_state[model_name]['status'] = f'Error: {str(e)}'
    finally:
        training_state[model_name]['running'] = False

@app.route('/train_status/<model_name>')
def train_status(model_name):
    """Get training status"""
    if model_name not in ['model1', 'model2']:
        return jsonify({'error': 'Invalid model name'}), 400
    
    return jsonify(training_state[model_name])

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on drawn digit"""
    try:
        data = request.json
        image_data = data.get('image')
        model_name = data.get('model', 'model1')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        if not check_model_exists(model_name):
            return jsonify({'error': f'{model_name} is not trained yet. Please train the model first.'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Make prediction based on model
        if model_name == 'model1':
            import joblib
            model = joblib.load('model1.joblib')
            
            # Flatten for sklearn
            img_flat = img_array.reshape(1, -1)
            prediction = model.predict(img_flat)[0]
            probabilities = model.predict_proba(img_flat)[0].tolist()
            
        elif model_name == 'model2':
            import tensorflow as tf
            model = tf.keras.models.load_model('model2.keras')
            
            # Add channel dimension for CNN
            img_array = img_array.reshape(1, 28, 28, 1)
            probs = model.predict(img_array, verbose=0)
            prediction = int(np.argmax(probs[0]))
            probabilities = probs[0].tolist()
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('trainers', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, port=5000)
