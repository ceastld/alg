from typing import List, Tuple
import math


class LogisticRegression:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.n_weights = n_features + 1
        self.weights = [0.0] * self.n_weights

    def sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def forward_pass(self, features: List[float]) -> Tuple[float, float]:
        z = self.weights[0]
        for i in range(self.n_features):
            z += self.weights[i + 1] * features[i]
        return z, self.sigmoid(z)

    def train(self, data: List[List[float]], learning_rate: float = 0.01, iterations: int = 100) -> None:
        for _ in range(iterations):
            for features, label in data:
                z, probability = self.forward_pass(features)
                error = probability - label
                for i in range(self.n_features):
                    self.weights[i + 1] -= learning_rate * error * features[i]
                self.weights[0] -= learning_rate * error

    def predict_probability(self, features: List[float]) -> float:
        return self.forward_pass(features)[1]

    def predict(self, features: List[float]) -> int:
        return 1 if self.predict_probability(features) >= 0.5 else 0

    def get_weights(self) -> List[float]:
        return self.weights


if __name__ == "__main__":
    model = LogisticRegression(n_features=2)
    data = [[[1.0, 2.0], 0], [[2.0, 1.0], 0], [[3.0, 3.0], 0], [[5.0, 4.0], 1], [[6.0, 5.0], 1], [[7.0, 6.0], 1]]
    model.train(data, learning_rate=0.1, iterations=1000)
    print(model.get_weights())
    print(model.predict([1.5, 1.5]))
    print(model.predict([5.5, 5.5]))
    print(model.predict([4.0, 4.0]))
