// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let brushSize = 10;

// Initialize canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = brushSize;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing functions
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictionOutput').style.display = 'none';
    document.getElementById('predictionError').style.display = 'none';
}

function invertCanvas() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
    }
    ctx.putImageData(imageData, 0, 0);
}

function updateBrushSize(size) {
    brushSize = parseInt(size);
    ctx.lineWidth = brushSize;
    document.getElementById('brushSizeLabel').textContent = brushSize;
}

// Prediction
async function predict() {
    const model = document.getElementById('modelSelect').value;
    const imageData = canvas.toDataURL('image/png');
    
    document.getElementById('predictionError').style.display = 'none';
    document.getElementById('predictionOutput').style.display = 'none';
    
    // Show loading state
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.textContent;
    predictBtn.textContent = '🔄 Predicting...';
    predictBtn.disabled = true;
    
    try {
        console.log('Sending prediction request for model:', model);
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                model: model
            })
        });
        
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        displayPrediction(data.prediction, data.probabilities);
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('predictionError').textContent = error.message;
        document.getElementById('predictionError').style.display = 'block';
        document.getElementById('predictionOutput').style.display = 'none';
    } finally {
        predictBtn.textContent = originalText;
        predictBtn.disabled = false;
    }
}

function displayPrediction(digit, probabilities) {
    document.getElementById('predictionDigit').textContent = digit;
    
    const probsContainer = document.getElementById('probabilities');
    probsContainer.innerHTML = '';
    
    probabilities.forEach((prob, idx) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        if (idx === digit) {
            probItem.style.borderColor = '#667eea';
            probItem.style.background = '#f0f4ff';
        }
        probItem.innerHTML = `
            <div class="prob-digit">${idx}</div>
            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
        `;
        probsContainer.appendChild(probItem);
    });
    
    document.getElementById('predictionOutput').style.display = 'block';
}

// Training
async function trainModel(modelName) {
    const btn = document.getElementById(`train${modelName.charAt(0).toUpperCase() + modelName.slice(1)}Btn`);
    btn.disabled = true;
    
    updateTrainingStatus(modelName, 'training');
    
    try {
        const response = await fetch(`/train/${modelName}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Training failed to start');
        }
        
        // Start polling for status
        pollTrainingStatus(modelName);
    } catch (error) {
        alert(`Error starting training: ${error.message}`);
        btn.disabled = false;
        updateTrainingStatus(modelName, 'untrained');
    }
}

function pollTrainingStatus(modelName) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/train_status/${modelName}`);
            const data = await response.json();
            
            // Update progress bar
            const progressNum = data.progress || 0;
            const progressEl = document.getElementById(`progress${modelName === 'model1' ? '1' : '2'}`);
            progressEl.style.width = `${progressNum}%`;
            progressEl.textContent = `${progressNum}%`;
            
            // Update status text
            const statusEl = document.getElementById(`status${modelName === 'model1' ? '1' : '2'}`);
            statusEl.textContent = data.status || '';
            
            // Update accuracy if available
            const accuracyEl = document.getElementById(`accuracy${modelName === 'model1' ? '1' : '2'}`);
            if (data.accuracy !== null && data.accuracy !== undefined) {
                accuracyEl.textContent = `Validation Accuracy: ${(data.accuracy * 100).toFixed(2)}%`;
            }
            
            // Check if training is complete
            if (!data.running) {
                clearInterval(interval);
                const btn = document.getElementById(`train${modelName.charAt(0).toUpperCase() + modelName.slice(1)}Btn`);
                btn.disabled = false;
                
                if (data.error) {
                    alert(`Training error: ${data.error}`);
                    updateTrainingStatus(modelName, 'untrained');
                } else {
                    updateTrainingStatus(modelName, 'ready');
                    checkModelStatus(); // Refresh model status
                }
            }
        } catch (error) {
            console.error('Error polling status:', error);
            clearInterval(interval);
        }
    }, 1000); // Poll every second
}

function updateTrainingStatus(modelName, status) {
    const badge = document.getElementById(`${modelName}Status`);
    badge.className = `status-badge ${status}`;
    
    if (status === 'ready') {
        badge.textContent = 'Ready';
    } else if (status === 'untrained') {
        badge.textContent = 'Untrained';
    } else if (status === 'training') {
        badge.textContent = 'Training...';
    }
}

// Check model status on load
async function checkModelStatus() {
    try {
        const response = await fetch('/model_status');
        const data = await response.json();
        
        // Update Model 1 status
        if (data.model1) {
            updateTrainingStatus('model1', 'ready');
            document.getElementById('status1').textContent = 'Model trained and ready';
        }
        
        // Update Model 2 status
        if (data.model2) {
            updateTrainingStatus('model2', 'ready');
            document.getElementById('status2').textContent = 'Model trained and ready';
        }
        
        // Disable Model 2 training if TensorFlow not available
        if (!data.tf_available) {
            const btn = document.getElementById('trainModel2Btn');
            btn.disabled = true;
            btn.textContent = 'TensorFlow Not Installed';
            document.getElementById('status2').textContent = 'TensorFlow not available. Install with: pip install tensorflow-macos';
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

// Initialize on page load
window.addEventListener('load', () => {
    checkModelStatus();
});
