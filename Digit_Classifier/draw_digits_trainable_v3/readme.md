# MNIST Digit Classifier Web App

A browser-based machine learning application for training and testing digit classification models on MNIST data. Draw digits in the browser and get real-time predictions from either a scikit-learn logistic regression model or a Keras CNN.

## Features

- **Interactive Canvas**: Draw digits with adjustable brush size, clear, and invert functions
- **Dual Model Support**: 
  - Model 1: Scikit-learn SGDClassifier (logistic regression)
  - Model 2: Keras CNN (Convolutional Neural Network)
- **Real-time Training**: Train models with live progress bars and status updates
- **Model Persistence**: Trained models are saved and automatically loaded for predictions
- **Apple Silicon Optimized**: Designed to run efficiently on macOS M1/M2/M3 chips

## Project Structure

```
.
├── app.py                      # Flask backend server
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── trainers/
│   ├── model1_trainer.py      # Scikit-learn model trainer
│   └── model2_trainer.py      # Keras CNN model trainer
├── templates/
│   └── index.html             # Main web interface
├── static/
│   └── app.js                 # Frontend JavaScript
├── model1.joblib              # (Generated) Trained Model 1
└── model2.keras               # (Generated) Trained Model 2
```

## Setup Instructions

### 1. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Base Requirements

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web server)
- NumPy (numerical computing)
- Pillow (image processing)
- Scikit-learn (Model 1)
- Joblib (model serialization)

### 3. (Optional) Enable Model 2 - Install TensorFlow

For Apple Silicon Macs:

```bash
pip install tensorflow-macos tensorflow-metal
```

For other platforms:

```bash
pip install tensorflow
```

**Note**: The app will work without TensorFlow, but Model 2 training will be disabled.

## Running the Application

### Start the Server

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`

### Open in Browser

Navigate to `http://127.0.0.1:5000` in your web browser.

## Usage Guide

### Training Models

1. **Model 1 (Scikit-learn)**:
   - Click "Train Model 1" button
   - Training takes ~2-5 minutes
   - Progress updates in real-time
   - Final validation accuracy displayed when complete

2. **Model 2 (Keras CNN)**:
   - Requires TensorFlow installation
   - Click "Train Model 2" button
   - Training takes ~5-10 minutes (includes GPU acceleration on Apple Silicon)
   - Progress updates per epoch
   - Higher accuracy than Model 1 (~98-99%)

### Making Predictions

1. Draw a digit (0-9) on the canvas
2. Select your desired model from the dropdown
3. Click "Predict" button
4. View the prediction and probability distribution

### Canvas Controls

- **Clear**: Erase the canvas
- **Invert**: Swap black/white (useful for different drawing styles)
- **Brush Size**: Adjust drawing thickness (5-30 pixels)

## API Endpoints

### Frontend Routes

- `GET /` - Main application page
- `GET /env` - Environment information (Python version, architecture, TF availability)
- `GET /model_status` - Check if models are trained

### Training Routes

- `POST /train/model1` - Start Model 1 training
- `POST /train/model2` - Start Model 2 training
- `GET /train_status/model1` - Get Model 1 training progress
- `GET /train_status/model2` - Get Model 2 training progress

### Prediction Route

- `POST /predict` - Make prediction on drawn digit
  - Body: `{"image": "data:image/png;base64,...", "model": "model1" or "model2"}`
  - Returns: `{"prediction": 5, "probabilities": [0.01, 0.02, ..., 0.85, ...]}`

## Training Progress Format

The trainer scripts output progress in JSON format via stdout:

```
PROGRESS:{"progress": 45, "status": "Epoch 3/10, Batch 5/12", "accuracy": 0.9234}
```

The Flask backend parses this and exposes it via the `/train_status` endpoint.

## Customizing Models

### Swapping in Custom Model Code

You can replace the default model implementations by modifying the trainer files:

#### Option 1: Edit Trainer Files Directly

Modify `trainers/model1_trainer.py` or `trainers/model2_trainer.py` with your custom model code. Ensure you:
- Keep the progress reporting calls: `report_progress(progress, status, accuracy)`
- Save to the correct filenames: `model1.joblib` or `model2.keras`
- Handle the MNIST data loading and preprocessing

#### Option 2: Create Wrapper Scripts

If you have existing `Model1.py` or `Model2.py` files without CLI support, create thin wrapper scripts:

**Example wrapper for custom Model1.py:**

```python
# trainers/model1_trainer.py
import sys
sys.path.insert(0, '/path/to/your/models')

from Model1 import YourModel  # Your custom model class
from sklearn.datasets import fetch_openml
import joblib
import json

def report_progress(progress, status, accuracy=None):
    data = {'progress': progress, 'status': status}
    if accuracy is not None:
        data['accuracy'] = accuracy
    print(f'PROGRESS:{json.dumps(data)}', flush=True)

# Load MNIST
report_progress(10, 'Loading data...')
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

# Initialize and train your model
report_progress(20, 'Training...')
model = YourModel()
model.fit(X, y)  # Adapt to your model's API

# Save
report_progress(90, 'Saving...')
joblib.dump(model, 'model1.joblib')
report_progress(100, 'Done', accuracy=0.95)
```

### Model Requirements

For the prediction endpoint to work:

**Model 1** must:
- Support `predict()` and `predict_proba()` methods
- Accept flattened 28×28 images (784 features)
- Be serializable with `joblib`

**Model 2** must:
- Be a Keras model
- Accept 28×28×1 input shape
- Output 10 class probabilities
- Be saveable with `.save()` method

## Model Performance

### Expected Accuracies

- **Model 1 (SGDClassifier)**: ~91-93% validation accuracy
- **Model 2 (Keras CNN)**: ~98-99% validation accuracy

### Training Times (Apple M1 Pro)

- **Model 1**: 2-3 minutes (CPU)
- **Model 2**: 5-7 minutes (GPU accelerated via Metal)

## Troubleshooting

### TensorFlow Not Found

If you see "TensorFlow not installed" for Model 2:
```bash
pip install tensorflow-macos tensorflow-metal  # Apple Silicon
# or
pip install tensorflow  # Other platforms
```

### Training Hangs

- Check terminal for error messages
- Ensure sufficient disk space for MNIST dataset (~50MB)
- Model 1 needs ~2GB RAM, Model 2 needs ~4GB RAM

### Prediction Returns "Model not trained"

- Train the model first by clicking the Train button
- Check that `model1.joblib` or `model2.keras` exists in the project root
- Verify no errors occurred during training

### Canvas Not Responding

- Try refreshing the page
- Check browser console for JavaScript errors
- Ensure JavaScript is enabled

### Low Accuracy

- For drawing: Try drawing larger, centered digits
- The canvas inverts black/white, so white digit on black works best
- Brush size 10-15px typically works well

## Environment Information

Check your environment with:
```bash
curl http://127.0.0.1:5000/env
```

Returns:
```json
{
  "python": "3.11.x",
  "arch": "arm64",
  "tf_available": true
}
```

## Development

### Running in Debug Mode

The app runs in debug mode by default:
```python
app.run(debug=True, port=5000)
```

### Adding New Models

To add a third model (Model 3):

1. Create `trainers/model3_trainer.py`
2. Add training state in `app.py`: `'model3': {...}`
3. Add `/train/model3` route
4. Update HTML to include Model 3 UI
5. Save model as `model3.pkl` (or appropriate extension)
6. Update predict endpoint to handle Model 3

## Technical Details

### Image Preprocessing Pipeline

1. Canvas captures 280×280 drawing
2. Sent as base64 PNG to backend
3. Converted to grayscale
4. Resized to 28×28 using Lanczos resampling
5. Normalized to [0, 1] range
6. For Model 1: Flattened to 784 features
7. For Model 2: Reshaped to (1, 28, 28, 1)

### Training Architecture

- **Model 1**: SGD with log loss, adaptive learning rate, mini-batch training
- **Model 2**: Conv(32)→Pool→Conv(64)→Pool→Dense(128)→Dropout(0.5)→Dense(10)

### Concurrency

- Training runs in background threads (one per model)
- Multiple training sessions cannot run simultaneously for the same model
- Predictions can be made while training is in progress

## License

This project is provided as-is for educational purposes.

## Credits

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- Built with Flask, Scikit-learn, TensorFlow/Keras
