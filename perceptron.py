import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, no_input, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight = np.random.normal(0, 0.5, size=(no_input + 1,))
        self.error = []
        self.labels = []

    def activation_function(self, value, name='unit_step'):
        if name == 'unit_step':
            return np.where(value >= 0, 1, 0)
        elif name == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif name == 'tanh':
            return np.tanh(value)
        elif name == 'relu':
            return np.maximum(0, value)
        else:
            return value

    def learning(self, inputs, act_func='unit_step'):
        summation = np.dot(inputs, self.weight[1:]) + self.weight[0]
        return self.activation_function(summation, act_func)

    def training(self, X, y, act_func='unit_step'):
        self.labels = sorted(set(y))
        for _ in range(self.epochs):
            err_sum = 0
            for inputs, label in zip(X, y):
                prediction = self.learning(inputs, act_func)
                update = self.learning_rate * (label - prediction)
                self.weight[1:] += update * inputs
                self.weight[0] += update
                err_sum += abs(label - prediction)
            self.error.append(err_sum)
        return self.error

    def predict(self, X, act_func='unit_step'):
        results = []
        for x in X:
            results.append(self.learning(x, act_func))
        return np.array(results)

# === Example: Test on Linear vs Nonlinear Problem ===
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])  # linear separable

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])  # non-linear

p = Perceptron(no_input=2, epochs=20, learning_rate=0.1)

print("=== AND Problem ===")
p.training(X_and, y_and)
print("Predictions:", p.predict(X_and))
plt.plot(p.error)
plt.title("Error over epochs (AND)")
plt.show()

print("\n=== XOR Problem ===")
p2 = Perceptron(no_input=2, epochs=20, learning_rate=0.1)
p2.training(X_xor, y_xor)
print("Predictions:", p2.predict(X_xor))
plt.plot(p2.error)
plt.title("Error over epochs (XOR)")
plt.show()
