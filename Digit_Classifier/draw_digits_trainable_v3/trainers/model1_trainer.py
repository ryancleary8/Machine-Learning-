"""
Model1 Trainer: Scikit-learn SGDClassifier (logistic regression)
Trains on MNIST with mini-batch partial_fit and progress reporting
"""
import json
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def report_progress(progress, status, accuracy=None):
    """Report progress in JSON format for parent process to parse"""
    data = {'progress': progress, 'status': status}
    if accuracy is not None:
        data['accuracy'] = accuracy
    print(f'PROGRESS:{json.dumps(data)}', flush=True)

def main():
    try:
        report_progress(5, 'Loading MNIST dataset...')
        
        # Load MNIST - explicitly use pandas parser
        try:
            mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        except Exception as e:
            # Fallback: try with explicit pandas parser
            mnist = fetch_openml('mnist_784', version=1, parser='pandas', as_frame=False)
        X, y = mnist.data, mnist.target
        
        # Convert to numpy arrays
        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='int')
        
        # Normalize
        X = X / 255.0
        
        report_progress(15, 'Splitting dataset...')
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        report_progress(20, 'Initializing model...')
        
        # Create SGDClassifier with log loss (logistic regression)
        model = SGDClassifier(
            loss='log_loss',
            max_iter=1,
            tol=None,
            random_state=42,
            learning_rate='adaptive',
            eta0=0.01,
            warm_start=True
        )
        
        # Train in mini-batches
        batch_size = 5000
        n_epochs = 20
        n_batches = len(X_train) // batch_size
        
        report_progress(25, f'Training with {n_epochs} epochs, {n_batches} batches per epoch...')
        
        best_accuracy = 0
        classes = np.unique(y_train)
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Partial fit
                model.partial_fit(X_batch, y_batch, classes=classes)
                
                # Calculate progress
                total_batches = n_epochs * n_batches
                current_batch = epoch * n_batches + batch_idx + 1
                progress = 25 + int((current_batch / total_batches) * 65)
                
                status = f'Epoch {epoch+1}/{n_epochs}, Batch {batch_idx+1}/{n_batches}'
                report_progress(progress, status)
            
            # Evaluate after each epoch
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            best_accuracy = max(best_accuracy, accuracy)
            
            report_progress(
                25 + int(((epoch + 1) / n_epochs) * 65),
                f'Epoch {epoch+1}/{n_epochs} complete - Val accuracy: {accuracy:.4f}'
            )
        
        report_progress(90, 'Evaluating final model...')
        
        # Final evaluation
        y_pred = model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        report_progress(95, 'Saving model...')
        
        # Save model
        joblib.dump(model, 'model1.joblib')
        
        report_progress(100, f'Training complete - Final validation accuracy: {final_accuracy:.4f}', final_accuracy)
        
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
