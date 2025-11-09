"""
Model2 Trainer: Keras CNN
Trains on MNIST with progress reporting via custom callback
"""
import json
import sys
import numpy as np

def report_progress(progress, status, accuracy=None):
    """Report progress in JSON format for parent process to parse"""
    data = {'progress': progress, 'status': status}
    if accuracy is not None:
        data['accuracy'] = accuracy
    print(f'PROGRESS:{json.dumps(data)}', flush=True)

def main():
    try:
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            print('ERROR: TensorFlow not installed', file=sys.stderr, flush=True)
            sys.exit(1)
        
        report_progress(5, 'Loading MNIST dataset...')
        
        # Load MNIST from Keras
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        report_progress(15, 'Preprocessing data...')
        
        # Normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        report_progress(20, 'Building CNN model...')
        
        # Build CNN model
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        report_progress(25, 'Starting training...')
        
        # Custom callback for progress reporting
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                self.best_val_acc = 0
            
            def on_epoch_begin(self, epoch, logs=None):
                progress = 25 + int((epoch / self.total_epochs) * 65)
                report_progress(progress, f'Training epoch {epoch+1}/{self.total_epochs}...')
            
            def on_epoch_end(self, epoch, logs=None):
                val_acc = logs.get('val_accuracy', 0)
                self.best_val_acc = max(self.best_val_acc, val_acc)
                progress = 25 + int(((epoch + 1) / self.total_epochs) * 65)
                report_progress(
                    progress,
                    f'Epoch {epoch+1}/{self.total_epochs} complete - Val accuracy: {val_acc:.4f}',
                    val_acc
                )
        
        n_epochs = 10
        callback = ProgressCallback(n_epochs)
        
        # Train model
        model.fit(
            X_train, y_train,
            batch_size=128,
            epochs=n_epochs,
            validation_data=(X_test, y_test),
            callbacks=[callback],
            verbose=0
        )
        
        report_progress(90, 'Evaluating final model...')
        
        # Final evaluation
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        report_progress(95, 'Saving model...')
        
        # Save model
        model.save('model2.keras')
        
        report_progress(100, f'Training complete - Final validation accuracy: {test_accuracy:.4f}', test_accuracy)
        
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
