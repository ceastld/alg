import math
from typing import List, Optional


class DeviceFailurePredictor:
    """Device failure prediction using logistic regression"""
    
    def __init__(self):
        self.weights = [0.0] * 6  # w0, w1, w2, w3, w4, w5
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
    
    def clean_data(self, raw_data: List[List[str]]) -> List[List[float]]:
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
    
    def fill_data(self, data: List[List[float]]) -> List[List[float]]:
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
    
    def sigmoid(self, z: float) -> float:
        """Sigmoid function"""
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0
    
    def predict_prob(self, features: List[float]) -> float:
        """Predict probability"""
        z = self.weights[0] + sum(self.weights[i+1] * features[i] for i in range(5))
        return self.sigmoid(z)
    
    def train(self, data: List[List[float]], lr: float = 0.01, iters: int = 100) -> None:
        """Train model using batch gradient descent"""
        if not data:
            return
        n = len(data)
        self.weights = [0.0] * 6
        
        for _ in range(iters):
            gradients = [0.0] * 6
            for sample in data:
                features, label = sample[:5], sample[5]
                prob = self.predict_prob(features)
                error = prob - label
                gradients[0] += error
                for i in range(5):
                    gradients[i+1] += error * features[i]
            for i in range(6):
                self.weights[i] -= lr * gradients[i] / n
    
    def predict(self, features: List[float]) -> int:
        """Predict binary classification"""
        return 1 if self.predict_prob(features) >= 0.5 else 0


def main():
    """Main function to handle input/output"""
    predictor = DeviceFailurePredictor()
    
    # Read training data
    n = int(input().strip())
    training_raw = [input().strip().split(',') for _ in range(n)]
    
    # Clean and prepare training data
    training_cleaned = predictor.clean_data(training_raw)
    training_final = predictor.fill_data(training_cleaned)
    
    # Train model
    predictor.train(training_final)
    
    # Read test data
    m = int(input().strip())
    test_data = []
    for _ in range(m):
        line = input().strip().split(',')
        features = [predictor.parse_float(line[i+1]) for i in range(5)]
        test_data.append(features)
    
    # Fill missing values and outliers in test data
    for sample in test_data:
        for idx in range(5):
            value = sample[idx]
            if value is None:
                sample[idx] = predictor.stats[idx][0]
            elif not predictor.is_valid(value, idx):
                sample[idx] = predictor.stats[idx][1]
    
    # Make predictions
    for sample in test_data:
        print(predictor.predict(sample))


if __name__ == "__main__":
    main()
