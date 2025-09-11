#!/usr/bin/env python3
"""
Example usage of the generic LogisticRegression class
"""

from 250903_1 import LogisticRegression

def example_2d_classification():
    """Example with 2 features"""
    print("=== 2D Classification Example ===")
    
    # Create model with 2 features
    model = LogisticRegression(n_features=2)
    
    # Training data: [x1, x2, label]
    training_data = [
        [1.0, 2.0, 0],  # Class 0
        [2.0, 1.0, 0],  # Class 0
        [3.0, 3.0, 0],  # Class 0
        [5.0, 4.0, 1],  # Class 1
        [6.0, 5.0, 1],  # Class 1
        [7.0, 6.0, 1],  # Class 1
    ]
    
    # Train the model
    model.train(training_data, learning_rate=0.1, iterations=1000)
    
    # Test predictions
    test_features = [
        [1.5, 1.5],  # Should be class 0
        [5.5, 5.5],  # Should be class 1
        [4.0, 4.0],  # Boundary case
    ]
    
    print("Test predictions:")
    for features in test_features:
        prob = model.predict_probability(features)
        pred = model.predict(features)
        print(f"Features {features}: Probability={prob:.3f}, Prediction={pred}")
    
    print(f"Final weights: {model.get_weights()}")
    print()

def example_3d_classification():
    """Example with 3 features"""
    print("=== 3D Classification Example ===")
    
    # Create model with 3 features
    model = LogisticRegression(n_features=3)
    
    # Training data: [x1, x2, x3, label]
    training_data = [
        [1.0, 2.0, 1.0, 0],  # Class 0
        [2.0, 1.0, 1.5, 0],  # Class 0
        [1.5, 1.5, 1.2, 0],  # Class 0
        [5.0, 4.0, 3.0, 1],  # Class 1
        [6.0, 5.0, 4.0, 1],  # Class 1
        [5.5, 4.5, 3.5, 1],  # Class 1
    ]
    
    # Train the model
    model.train(training_data, learning_rate=0.1, iterations=1000)
    
    # Test predictions
    test_features = [
        [1.2, 1.8, 1.1],  # Should be class 0
        [5.8, 4.8, 3.8],  # Should be class 1
        [3.0, 3.0, 2.0],  # Boundary case
    ]
    
    print("Test predictions:")
    for features in test_features:
        prob = model.predict_probability(features)
        pred = model.predict(features)
        print(f"Features {features}: Probability={prob:.3f}, Prediction={pred}")
    
    print(f"Final weights: {model.get_weights()}")
    print()

def example_custom_threshold():
    """Example with custom decision threshold"""
    print("=== Custom Threshold Example ===")
    
    model = LogisticRegression(n_features=2)
    
    # Simple training data
    training_data = [
        [1.0, 1.0, 0],
        [2.0, 2.0, 0],
        [4.0, 4.0, 1],
        [5.0, 5.0, 1],
    ]
    
    model.train(training_data, learning_rate=0.1, iterations=1000)
    
    test_features = [3.0, 3.0]  # Boundary case
    
    print(f"Test features: {test_features}")
    prob = model.predict_probability(test_features)
    print(f"Probability: {prob:.3f}")
    
    # Different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        pred = model.predict(test_features, threshold=threshold)
        print(f"Threshold {threshold}: Prediction={pred}")

if __name__ == "__main__":
    example_2d_classification()
    example_3d_classification()
    example_custom_threshold()
