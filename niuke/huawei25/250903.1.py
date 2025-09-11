import math
from typing import List, Optional, Tuple


class DataProcessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.stats = []  # (mean, median) for each feature
        
    def parse_float(self, value: str) -> Optional[float]:
        """Parse string to float, return None if invalid"""
        if value.strip() == 'NaN':
            return None
        try:
            return float(value.strip())
        except ValueError:
            return None
    
    def is_valid(self, value: float, idx: int) -> bool:
        """Check if value is within valid range"""
        ranges = [
            (0, float('inf')),      # writes
            (0, float('inf')),      # reads  
            (0, 1000),              # avg_write_ms
            (0, 1000),              # avg_read_ms
            (0, 20)                 # years
        ]
        return ranges[idx][0] <= value <= ranges[idx][1]
    
    def calculate_stats(self, data: List[List[float]]) -> None:
        """Calculate mean and median for each feature"""
        self.stats = []
        for idx in range(5):
            valid = [sample[idx] for sample in data 
                    if sample[idx] is not None and self.is_valid(sample[idx], idx)]
            if valid:
                mean_val = sum(valid) / len(valid)
                valid.sort()
                n = len(valid)
                median_val = (valid[n//2-1] + valid[n//2]) / 2 if n % 2 == 0 else valid[n//2]
            else:
                mean_val = median_val = 0.0
            self.stats.append((mean_val, median_val))
    
    def clean_training_data(self, raw_data: List[List[str]]) -> List[List[float]]:
        """Clean training data"""
        cleaned = []
        for row in raw_data:
            if len(row) < 7:
                continue
            status = self.parse_float(row[6])
            if status not in [0.0, 1.0]:
                continue
            features = [self.parse_float(row[i+1]) for i in range(5)]
            features.append(status)
            cleaned.append(features)
        return cleaned
    
    def fill_missing_and_outliers(self, data: List[List[float]]) -> List[List[float]]:
        """Fill missing values and outliers"""
        if not data:
            return data
        self.calculate_stats(data)
        for sample in data:
            for idx in range(5):
                value = sample[idx]
                if value is None:
                    sample[idx] = self.stats[idx][0]  # mean
                elif not self.is_valid(value, idx):
                    sample[idx] = self.stats[idx][1]  # median
        return data
    
    def process_test_data(self, raw_data: List[List[str]]) -> List[List[float]]:
        """Process test data using training statistics"""
        test_data = []
        for row in raw_data:
            features = [self.parse_float(row[i+1]) for i in range(5)]
            test_data.append(features)
        
        # Fill missing values and outliers using training stats
        for sample in test_data:
            for idx in range(5):
                value = sample[idx]
                if value is None:
                    sample[idx] = self.stats[idx][0]
                elif not self.is_valid(value, idx):
                    sample[idx] = self.stats[idx][1]
        return test_data


class LogisticRegression:
    """Generic logistic regression model with clear gradient descent steps"""
    
    def __init__(self, n_features: int):
        """
        Initialize logistic regression model
        
        Args:
            n_features: Number of input features
        """
        self.n_features = n_features
        self.n_weights = n_features + 1  # +1 for bias term
        self.weights = [0.0] * self.n_weights
    
    def sigmoid(self, z: float) -> float:
        """Sigmoid activation function"""
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0
    
    def forward_pass(self, features: List[float]) -> Tuple[float, float]:
        """
        Forward pass: compute linear combination and probability
        
        Args:
            features: Input features [x1, x2, ..., xn]
            
        Returns:
            Tuple of (linear_combination, probability)
        """
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
        
        # Linear combination: z = w0 + w1*x1 + w2*x2 + ... + wn*xn
        z = self.weights[0]  # bias term
        for i in range(self.n_features):
            z += self.weights[i+1] * features[i]
        
        # Apply sigmoid activation: P(y=1) = σ(z)
        probability = self.sigmoid(z)
        
        return z, probability
    
    def compute_gradients(self, data: List[List[float]]) -> List[float]:
        """
        Compute gradients for all weights
        
        Args:
            data: Training data where each sample is [x1, x2, ..., xn, label]
            
        Returns:
            List of average gradients for all weights
        """
        n_samples = len(data)
        gradients = [0.0] * self.n_weights
        
        for sample in data:
            features = sample[:-1]  # All except last element
            label = sample[-1]      # Last element is label
            
            if len(features) != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
            
            # Forward pass
            z, probability = self.forward_pass(features)
            
            # Compute prediction error: error = P(y=1) - y
            error = probability - label
            
            # Compute gradients
            # ∂L/∂w0 = error (bias gradient)
            gradients[0] += error
            
            # ∂L/∂wi = error * xi (feature gradients)
            for i in range(self.n_features):
                gradients[i+1] += error * features[i]
        
        # Return average gradients
        return [grad / n_samples for grad in gradients]
    
    def update_weights(self, gradients: List[float], learning_rate: float) -> None:
        """
        Update weights using gradient descent
        
        Args:
            gradients: List of gradients for all weights
            learning_rate: Learning rate for weight updates
        """
        if len(gradients) != self.n_weights:
            raise ValueError(f"Expected {self.n_weights} gradients, got {len(gradients)}")
        
        for i in range(self.n_weights):
            self.weights[i] -= learning_rate * gradients[i]
    
    def train(self, data: List[List[float]], learning_rate: float = 0.01, iterations: int = 100) -> None:
        """
        Train model using batch gradient descent with clear steps
        
        Args:
            data: Training data where each sample is [x1, x2, ..., xn, label]
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
        """
        if not data:
            return
        
        # Validate data format
        for i, sample in enumerate(data):
            if len(sample) != self.n_features + 1:
                raise ValueError(f"Sample {i}: expected {self.n_features + 1} elements, got {len(sample)}")
        
        # Initialize weights to zero
        self.weights = [0.0] * self.n_weights
        
        # Training loop
        for iteration in range(iterations):
            # Step 1: Forward pass and compute gradients
            gradients = self.compute_gradients(data)
            
            # Step 2: Update weights
            self.update_weights(gradients, learning_rate)
    
    def predict_probability(self, features: List[float]) -> float:
        """
        Predict probability of positive class
        
        Args:
            features: Input features [x1, x2, ..., xn]
            
        Returns:
            Probability of positive class (0-1)
        """
        _, probability = self.forward_pass(features)
        return probability
    
    def predict(self, features: List[float], threshold: float = 0.5) -> int:
        """
        Predict binary classification
        
        Args:
            features: Input features [x1, x2, ..., xn]
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Binary prediction (0 or 1)
        """
        probability = self.predict_probability(features)
        return 1 if probability >= threshold else 0
    
    def get_weights(self) -> List[float]:
        """Get current model weights"""
        return self.weights.copy()
    
    def set_weights(self, weights: List[float]) -> None:
        """Set model weights"""
        if len(weights) != self.n_weights:
            raise ValueError(f"Expected {self.n_weights} weights, got {len(weights)}")
        self.weights = weights.copy()


class DeviceFailurePredictor:
    """Main predictor class that combines data processing and model"""
    
    def __init__(self, n_features: int = 5):
        """
        Initialize device failure predictor
        
        Args:
            n_features: Number of input features (default 5 for device data)
        """
        self.n_features = n_features
        self.data_processor = DataProcessor()
        self.model = LogisticRegression(n_features)
    
    def train(self, raw_training_data: List[List[str]]) -> None:
        """Train the model with raw training data"""
        # Step 1: Clean training data
        cleaned_data = self.data_processor.clean_training_data(raw_training_data)
        
        # Step 2: Fill missing values and outliers
        processed_data = self.data_processor.fill_missing_and_outliers(cleaned_data)
        
        # Step 3: Train the model
        self.model.train(processed_data)
    
    def predict(self, raw_test_data: List[List[str]]) -> List[int]:
        """Predict on raw test data"""
        # Process test data using training statistics
        test_features = self.data_processor.process_test_data(raw_test_data)
        
        # Make predictions
        predictions = []
        for features in test_features:
            prediction = self.model.predict(features)
            predictions.append(prediction)
        
        return predictions
    
    def predict_with_probability(self, raw_test_data: List[List[str]]) -> List[Tuple[int, float]]:
        """Predict with probability scores"""
        test_features = self.data_processor.process_test_data(raw_test_data)
        
        predictions = []
        for features in test_features:
            probability = self.model.predict_probability(features)
            prediction = self.model.predict(features)
            predictions.append((prediction, probability))
        
        return predictions


def main():
    """Main function to handle input/output"""
    predictor = DeviceFailurePredictor()
    
    # Read training data
    n = int(input().strip())
    training_raw = [input().strip().split(',') for _ in range(n)]
    
    # Train model
    predictor.train(training_raw)
    
    # Read test data
    m = int(input().strip())
    test_raw = [input().strip().split(',') for _ in range(m)]
    
    # Make predictions
    predictions = predictor.predict(test_raw)
    for prediction in predictions:
        print(prediction)


if __name__ == "__main__":
    main()
