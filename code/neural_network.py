"""
Neural Network Module - Week 11 Requirement
Implements LSTM for time-series flood prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class FloodLSTM:
    """
    LSTM Neural Network for flood prediction using time-series data.
    Predicts flood probability based on sequential weather patterns.
    
    Architecture:
    - Input Layer: sequence_length x num_features
    - LSTM Layer 1: hidden_size units with tanh activation
    - LSTM Layer 2: hidden_size//2 units
    - Dense Layer: 1 unit with sigmoid activation
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 sequence_length: int = 7, learning_rate: float = 0.01):
        """
        Initialize LSTM network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            sequence_length: Number of time steps to look back
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self._initialize_weights()
        
        # Scalers for normalization
        self.feature_scaler = MinMaxScaler()
        self.is_fitted = False
        
    def _initialize_weights(self):
        """Initialize LSTM weights using Xavier initialization"""
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # LSTM Layer 1 weights
        # Gates: forget, input, output, cell candidate (4 gates)
        self.Wf1 = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wi1 = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wo1 = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wc1 = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        
        self.bf1 = np.zeros((self.hidden_size, 1))
        self.bi1 = np.zeros((self.hidden_size, 1))
        self.bo1 = np.zeros((self.hidden_size, 1))
        self.bc1 = np.zeros((self.hidden_size, 1))
        
        # Output layer weights
        self.Wy = np.random.randn(1, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        self.by = np.zeros((1, 1))
        
        # Store gradients
        self.gradients = {}
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(x)
    
    def _sigmoid_derivative(self, s: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid"""
        return s * (1 - s)
    
    def _tanh_derivative(self, t: np.ndarray) -> np.ndarray:
        """Derivative of tanh"""
        return 1 - t ** 2
    
    def _lstm_cell_forward(self, x_t: np.ndarray, h_prev: np.ndarray, 
                          c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass for a single LSTM cell.
        
        Args:
            x_t: Input at time t (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            c_prev: Previous cell state (hidden_size, 1)
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
            cache: Values needed for backpropagation
        """
        # Concatenate input and previous hidden state
        concat = np.vstack([h_prev, x_t])
        
        # Forget gate
        f_t = self._sigmoid(np.dot(self.Wf1, concat) + self.bf1)
        
        # Input gate
        i_t = self._sigmoid(np.dot(self.Wi1, concat) + self.bi1)
        
        # Candidate cell state
        c_tilde = self._tanh(np.dot(self.Wc1, concat) + self.bc1)
        
        # New cell state
        c_next = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = self._sigmoid(np.dot(self.Wo1, concat) + self.bo1)
        
        # New hidden state
        h_next = o_t * self._tanh(c_next)
        
        cache = {
            'concat': concat,
            'f_t': f_t, 'i_t': i_t, 'c_tilde': c_tilde, 'o_t': o_t,
            'c_prev': c_prev, 'c_next': c_next, 'h_prev': h_prev, 'h_next': h_next,
            'x_t': x_t
        }
        
        return h_next, c_next, cache
    
    def _forward_pass(self, X_seq: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Forward pass through entire sequence.
        
        Args:
            X_seq: Input sequence (sequence_length, input_size)
            
        Returns:
            y_pred: Predicted probability
            caches: List of caches for each time step
        """
        T = X_seq.shape[0]  # sequence length
        
        # Initialize hidden and cell states
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        caches = []
        
        # Process each time step
        for t in range(T):
            x_t = X_seq[t].reshape(-1, 1)
            h, c, cache = self._lstm_cell_forward(x_t, h, c)
            caches.append(cache)
        
        # Output layer - use final hidden state
        y_linear = np.dot(self.Wy, h) + self.by
        y_pred = self._sigmoid(y_linear)
        
        return y_pred, caches
    
    def _backward_pass(self, y_pred: np.ndarray, y_true: np.ndarray, 
                      caches: List[Dict]) -> Dict:
        """
        Backward pass through LSTM.
        
        Args:
            y_pred: Predicted probability
            y_true: True label
            caches: Caches from forward pass
            
        Returns:
            gradients: Dictionary of gradients
        """
        T = len(caches)
        
        # Initialize gradients
        dWf1 = np.zeros_like(self.Wf1)
        dWi1 = np.zeros_like(self.Wi1)
        dWo1 = np.zeros_like(self.Wo1)
        dWc1 = np.zeros_like(self.Wc1)
        dbf1 = np.zeros_like(self.bf1)
        dbi1 = np.zeros_like(self.bi1)
        dbo1 = np.zeros_like(self.bo1)
        dbc1 = np.zeros_like(self.bc1)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        
        # Output layer gradient
        dy = y_pred - y_true  # Binary cross-entropy derivative
        dWy = np.dot(dy, caches[-1]['h_next'].T)
        dby = dy
        
        # Gradient w.r.t final hidden state
        dh_next = np.dot(self.Wy.T, dy)
        dc_next = np.zeros((self.hidden_size, 1))
        
        # Backprop through time
        for t in reversed(range(T)):
            cache = caches[t]
            
            # Unpack cache
            f_t = cache['f_t']
            i_t = cache['i_t']
            c_tilde = cache['c_tilde']
            o_t = cache['o_t']
            c_prev = cache['c_prev']
            c_next = cache['c_next']
            concat = cache['concat']
            
            # Output gate gradients
            do_t = dh_next * self._tanh(c_next) * self._sigmoid_derivative(o_t)
            
            # Cell state gradients
            dc_next += dh_next * o_t * self._tanh_derivative(self._tanh(c_next))
            
            # Candidate cell gradients
            dc_tilde = dc_next * i_t * self._tanh_derivative(c_tilde)
            
            # Input gate gradients
            di_t = dc_next * c_tilde * self._sigmoid_derivative(i_t)
            
            # Forget gate gradients
            df_t = dc_next * c_prev * self._sigmoid_derivative(f_t)
            
            # Parameter gradients
            dWf1 += np.dot(df_t, concat.T)
            dWi1 += np.dot(di_t, concat.T)
            dWo1 += np.dot(do_t, concat.T)
            dWc1 += np.dot(dc_tilde, concat.T)
            dbf1 += df_t
            dbi1 += di_t
            dbo1 += do_t
            dbc1 += dc_tilde
            
            # Gradient w.r.t concat (for previous time step)
            dconcat = (np.dot(self.Wf1.T, df_t) + np.dot(self.Wi1.T, di_t) +
                      np.dot(self.Wo1.T, do_t) + np.dot(self.Wc1.T, dc_tilde))
            
            dh_next = dconcat[:self.hidden_size]
            dc_next = dc_next * f_t
        
        # Clip gradients to prevent explosion
        for grad in [dWf1, dWi1, dWo1, dWc1, dbf1, dbi1, dbo1, dbc1, dWy, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return {
            'dWf1': dWf1, 'dWi1': dWi1, 'dWo1': dWo1, 'dWc1': dWc1,
            'dbf1': dbf1, 'dbi1': dbi1, 'dbo1': dbo1, 'dbc1': dbc1,
            'dWy': dWy, 'dby': dby
        }
    
    def _update_weights(self, gradients: Dict):
        """Update weights using gradient descent"""
        self.Wf1 -= self.learning_rate * gradients['dWf1']
        self.Wi1 -= self.learning_rate * gradients['dWi1']
        self.Wo1 -= self.learning_rate * gradients['dWo1']
        self.Wc1 -= self.learning_rate * gradients['dWc1']
        self.bf1 -= self.learning_rate * gradients['dbf1']
        self.bi1 -= self.learning_rate * gradients['dbi1']
        self.bo1 -= self.learning_rate * gradients['dbo1']
        self.bc1 -= self.learning_rate * gradients['dbc1']
        self.Wy -= self.learning_rate * gradients['dWy']
        self.by -= self.learning_rate * gradients['dby']
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            X_seq: Sequences (n_sequences, sequence_length, n_features)
            y_seq: Targets for each sequence
        """
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            targets.append(y[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size (not used in basic implementation)
            verbose: Print training progress
            
        Returns:
            history: Training history with losses
        """
        # Normalize features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough samples to create sequences with length {self.sequence_length}")
        
        # Handle class imbalance with oversampling flood events
        flood_indices = np.where(y_seq == 1)[0]
        non_flood_indices = np.where(y_seq == 0)[0]
        
        if len(flood_indices) > 0 and len(flood_indices) < len(non_flood_indices):
            # Oversample flood events
            oversample_factor = min(5, len(non_flood_indices) // len(flood_indices))
            flood_indices_oversampled = np.tile(flood_indices, oversample_factor)
            all_indices = np.concatenate([non_flood_indices, flood_indices_oversampled])
        else:
            all_indices = np.arange(len(y_seq))
        
        history = {'loss': [], 'accuracy': []}
        
        print(f"\nTraining LSTM on {len(X_seq)} sequences...")
        print(f"Flood events: {sum(y_seq)}, Non-flood: {len(y_seq) - sum(y_seq)}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(all_indices)
            
            for idx in all_indices:
                # Forward pass
                y_pred, caches = self._forward_pass(X_seq[idx])
                
                # Calculate loss (binary cross-entropy)
                y_true = y_seq[idx].reshape(1, 1)
                epsilon = 1e-7
                loss = -y_true * np.log(y_pred + epsilon) - (1 - y_true) * np.log(1 - y_pred + epsilon)
                epoch_loss += loss[0, 0]
                
                # Backward pass
                gradients = self._backward_pass(y_pred, y_true, caches)
                
                # Update weights
                self._update_weights(gradients)
            
            avg_loss = epoch_loss / len(all_indices)
            history['loss'].append(avg_loss)
            
            # Calculate accuracy
            predictions = self.predict(X_scaled[self.sequence_length:], from_sequences=True)
            accuracy = accuracy_score(y_seq, predictions)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        self.is_fitted = True
        return history
    
    def predict_proba(self, X: np.ndarray, from_sequences: bool = False) -> np.ndarray:
        """
        Predict flood probabilities.
        
        Args:
            X: Features or pre-scaled features
            from_sequences: If True, X is already scaled
            
        Returns:
            probabilities: Predicted probabilities
        """
        if not from_sequences:
            X = self.feature_scaler.transform(X)
            X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        else:
            X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        
        probabilities = []
        for seq in X_seq:
            y_pred, _ = self._forward_pass(seq)
            probabilities.append(y_pred[0, 0])
        
        return np.array(probabilities)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5, 
                from_sequences: bool = False) -> np.ndarray:
        """
        Predict flood labels.
        
        Args:
            X: Features
            threshold: Classification threshold
            from_sequences: If True, X is already scaled
            
        Returns:
            predictions: Binary predictions
        """
        probabilities = self.predict_proba(X, from_sequences)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        X_scaled = self.feature_scaler.transform(X)
        _, y_seq = self.create_sequences(X_scaled, y)
        
        predictions = self.predict(X, threshold=0.5)
        probabilities = self.predict_proba(X)
        
        # Ensure we have matching lengths
        predictions = predictions[:len(y_seq)]
        probabilities = probabilities[:len(y_seq)]
        
        metrics = {
            'accuracy': accuracy_score(y_seq, predictions),
            'precision': precision_score(y_seq, predictions, zero_division=0),
            'recall': recall_score(y_seq, predictions, zero_division=0),
            'f1': f1_score(y_seq, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y_seq, predictions).tolist()
        }
        
        return metrics


class SimpleLSTMWrapper:
    """
    Wrapper for LSTM that provides sklearn-like interface.
    Uses simplified implementation suitable for demonstration.
    """
    
    def __init__(self, sequence_length: int = 7, hidden_size: int = 32):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> Dict:
        """Train the model"""
        self.model = FloodLSTM(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
            sequence_length=self.sequence_length,
            learning_rate=0.005
        )
        
        return self.model.fit(X, y, epochs=epochs, verbose=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict flood labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict flood probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model"""
        return self.model.evaluate(X, y)


def demo_lstm():
    """Demonstrate LSTM flood prediction"""
    print("=" * 60)
    print("LSTM NEURAL NETWORK FOR FLOOD PREDICTION - Demo")
    print("=" * 60)
    
    # Generate synthetic flood data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: temp, precipitation, humidity, pressure, wind
    X = np.random.randn(n_samples, 5)
    
    # Create flood labels based on high precipitation and humidity
    flood_probability = 0.3 * X[:, 1] + 0.2 * X[:, 2]  # precipitation + humidity
    y = (flood_probability > np.percentile(flood_probability, 90)).astype(int)
    
    print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
    print(f"Flood events: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    lstm = SimpleLSTMWrapper(sequence_length=7, hidden_size=32)
    history = lstm.fit(X_train, y_train, epochs=30)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    metrics = lstm.evaluate(X_test, y_test)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    
    return lstm, history, metrics


if __name__ == "__main__":
    demo_lstm()
