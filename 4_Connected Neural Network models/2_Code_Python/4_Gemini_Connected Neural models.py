import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # To generate multi-class data
# Suppress runtime warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Activation Functions ---
def softmax(z):
  """Softmax activation function for multi-class output"""
  # Shift z for numerical stability (subtract max value)
  exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
  return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def relu(z):
  """ReLU activation function"""
  return np.maximum(0, z)

# No need for softmax_derivative if using combined cross-entropy gradient
# def softmax_derivative(z): # Not typically needed directly in backprop like this
#   s = softmax(z)
#   # This is complex to compute element-wise, usually combined with loss derivative
#   pass

def relu_derivative(z):
  """Derivative of the ReLU function"""
  deriv = np.array(z, copy=True)
  deriv[deriv <= 0] = 0
  deriv[deriv > 0] = 1
  return deriv

# --- One-Hot Encoding ---
def to_one_hot(Y_indices, num_classes):
    """Converts a vector of class indices into a one-hot encoded matrix."""
    m = Y_indices.shape[0]
    Y_one_hot = np.zeros((num_classes, m))
    # Ensure Y_indices is flat for correct indexing
    Y_indices_flat = Y_indices.flatten().astype(int)
    Y_one_hot[Y_indices_flat, np.arange(m)] = 1
    return Y_one_hot

# --- Loss Function ---
def compute_multi_class_cost(Y_hat, Y_one_hot):
  """
  Computes the categorical cross-entropy cost.

  Arguments:
  Y_hat -- Output of the softmax activation (predictions), shape (num_classes, number of examples)
  Y_one_hot -- The true "label" vector as one-hot, shape (num_classes, number of examples)

  Returns:
  cost -- Cross-entropy cost
  """
  m = Y_one_hot.shape[1] # Number of examples

  # Add epsilon to prevent log(0) errors
  epsilon = 1e-8
  cost = - (1/m) * np.sum(Y_one_hot * np.log(Y_hat + epsilon))

  cost = np.squeeze(cost) # Ensure cost is scalar
  return cost

# --- Parameter Initialization ---
def initialize_parameters(n_x, n_h, n_y):
  """
  Initializes weights and biases for a 2-layer network. n_y is now num_classes.

  Arguments:
  n_x -- size of the input layer
  n_h -- size of the hidden layer
  n_y -- size of the output layer (number of classes)

  Returns:
  parameters -- python dictionary containing W1, b1, W2, b2
  """
  np.random.seed(2)
  W1 = np.random.randn(n_h, n_x) * 0.01 # Smaller scale often helps
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros((n_y, 1))
  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
  return parameters

# --- Forward Propagation ---
def forward_propagation(X, parameters):
  """
  Implements the forward propagation: LINEAR -> RELU -> LINEAR -> SOFTMAX

  Arguments:
  X -- input data of size (n_x, number of examples)
  parameters -- python dictionary containing W1, b1, W2, b2

  Returns:
  Y_hat -- The output probabilities from softmax, shape (n_y, m)
  cache -- a dictionary containing Z1, A1, Z2, A2 (=Y_hat)
  """
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']

  Z1 = np.dot(W1, X) + b1
  A1 = relu(Z1)
  Z2 = np.dot(W2, A1) + b2
  Y_hat = softmax(Z2) # Use softmax for output layer

  assert(Y_hat.shape == (W2.shape[0], X.shape[1]))

  cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": Y_hat}
  return Y_hat, cache

# --- Backward Propagation ---
def backward_propagation(parameters, cache, X, Y_one_hot):
  """
  Implements backward propagation for the LINEAR->RELU->LINEAR->SOFTMAX model
  using categorical cross-entropy loss.

  Arguments:
  parameters -- python dictionary containing W1, b1, W2, b2
  cache -- dictionary containing Z1, A1, Z2, A2 from forward prop
  X -- input data of shape (n_x, number of examples)
  Y_one_hot -- true "label" vector (one-hot), shape (n_y, number of examples)

  Returns:
  grads -- A dictionary with the gradients: dW1, db1, dW2, db2
  """
  m = X.shape[1]
  W1 = parameters['W1']
  W2 = parameters['W2']
  A1 = cache['A1']
  A2 = cache['A2'] # Y_hat
  Z1 = cache['Z1']

  # --- Backward propagation ---
  # Output Layer Gradients (using Softmax + Categorical Cross-Entropy simplified derivative)
  dZ2 = A2 - Y_one_hot # Shape (n_y, m)

  # Gradients for W2 and b2
  dW2 = (1/m) * np.dot(dZ2, A1.T) # Shape (n_y, n_h)
  db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) # Shape (n_y, 1)

  # Hidden Layer Gradients
  dA1 = np.dot(W2.T, dZ2) # Shape (n_h, m)
  dZ1 = dA1 * relu_derivative(Z1) # Shape (n_h, m)

  # Gradients for W1 and b1
  dW1 = (1/m) * np.dot(dZ1, X.T) # Shape (n_h, n_x)
  db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) # Shape (n_h, 1)

  grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
  return grads

# --- Update Parameters ---
def update_parameters(parameters, grads, learning_rate=0.001):
  """Updates parameters using gradient descent."""
  # Same logic as before, just using the calculated grads
  W1 = parameters['W1'] - learning_rate * grads['dW1']
  b1 = parameters['b1'] - learning_rate * grads['db1']
  W2 = parameters['W2'] - learning_rate * grads['dW2']
  b2 = parameters['b2'] - learning_rate * grads['db2']
  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
  return parameters

# --- Build the Neural Network Model ---
def nn_model_multi(X, Y_indices, n_h, num_classes, num_iterations=10000, learning_rate=0.001, print_cost=True):
  """
  Builds and trains the 2-layer neural network for multi-class classification.

  Arguments:
  X -- dataset of shape (n_x, number of examples)
  Y_indices -- vector of true labels (indices 0, 1, ..., K-1), shape (m,) or (1, m)
  n_h -- size of the hidden layer
  num_classes -- number of classes (K)
  num_iterations -- Number of iterations in gradient descent loop
  learning_rate -- Learning rate
  print_cost -- if True, print the cost every 1000 iterations

  Returns:
  parameters -- parameters learnt by the model
  costs -- list of costs recorded during training
  """
  np.random.seed(3)
  n_x = X.shape[0]
  n_y = num_classes # Output layer size is num_classes

  # Convert Y_indices to one-hot encoding
  Y_one_hot = to_one_hot(Y_indices, num_classes)

  # Initialize parameters
  parameters = initialize_parameters(n_x, n_h, n_y)

  costs = []

  # Gradient descent loop
  for i in range(0, num_iterations):
    # Forward propagation
    Y_hat, cache = forward_propagation(X, parameters)

    # Cost function
    cost = compute_multi_class_cost(Y_hat, Y_one_hot)

    # Backpropagation
    grads = backward_propagation(parameters, cache, X, Y_one_hot)

    # Gradient descent parameter update
    parameters = update_parameters(parameters, grads, learning_rate)

    # Record the cost
    if i % 100 == 0:
        costs.append(cost)

    if print_cost and i % 1000 == 0:
      print(f"Cost after iteration {i}: {cost}")

  return parameters, costs

# --- Prediction Function ---
def predict_multi(parameters, X):
  """
  Predicts the class for each example in X using learned parameters.

  Arguments:
  parameters -- python dictionary containing parameters W1, b1, W2, b2
  X -- input data of size (n_x, m)

  Returns
  predictions -- vector of predictions (class indices 0..K-1), shape (1, m)
  """
  Y_hat, cache = forward_propagation(X, parameters)
  # Get the index of the max probability
  predictions = np.argmax(Y_hat, axis=0)
  # Return as row vector
  return predictions.reshape(1, -1)

# --- Plotting Function (Modified for Multi-Class) ---
def plot_decision_boundary_multi(model, X, y_indices):
  """
  Plots the decision boundary for multi-class classification.

  Arguments:
  model -- a function that takes parameters and X, and returns prediction indices
  X -- input features, shape (n_x, m)
  y_indices -- true label indices, shape (1, m) or (m,)
  """
  # Ensure y_indices is flat for scatter plot coloring
  y_indices_flat = y_indices.flatten()

  x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
  y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
  h = 0.02 # step size in the mesh
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Predict the function value for the whole grid
  # model expects shape (n_x, m), so transpose the concatenated grid
  Z = model(np.c_[xx.ravel(), yy.ravel()].T)
  Z = Z.reshape(xx.shape)

  # Plot the contour and training examples
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
  plt.scatter(X[0, :], X[1, :], c=y_indices_flat, cmap=plt.cm.Spectral, edgecolors='k')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('Multi-Class Decision Boundary')
  plt.show()

# --- Main Execution ---

# 1. Generate Multi-Class Data
N_SAMPLES = 400 # Number of samples per class
N_CLASSES = 3 # Number of classes (e.g., 3 blobs)
X, y_indices = make_blobs(n_samples=N_SAMPLES, centers=N_CLASSES, n_features=2,
                          random_state=42, cluster_std=1.2)

# Transpose X to match our convention (features, samples)
X = X.T # Shape (2, N_SAMPLES)
# Ensure y_indices is a row vector (1, N_SAMPLES) or flat array (N_SAMPLES,)
y_indices = y_indices.reshape(1, -1) # Shape (1, N_SAMPLES)

# 2. Define Network Structure
n_x = X.shape[0]      # Input layer size (2 features)
n_h = 8               # Hidden layer size (tunable)
n_y = N_CLASSES       # Output layer size (number of classes)

# 3. Train the Model
learning_rate = 0.001
num_iterations = 1000
parameters, costs = nn_model_multi(X, y_indices, n_h=n_h, num_classes=n_y,
                                   num_iterations=num_iterations, learning_rate=learning_rate,
                                   print_cost=True)

# 4. Evaluate the Model (optional)
predictions = predict_multi(parameters, X)
accuracy = np.mean(predictions == y_indices) * 100
print(f'\nAccuracy on training set: {accuracy:.2f} %')

# 5. Visualize Results

# Plot cost function
plt.figure()
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title(f'Cost Reduction (LR={learning_rate})')
plt.show()

# Plot decision boundary
plt.figure(figsize=(8, 6))
# Create a lambda that takes only X, using the learned parameters
model_predictor = lambda x: predict_multi(parameters, x)
plot_decision_boundary_multi(model_predictor, X, y_indices)