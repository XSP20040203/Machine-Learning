import numpy as np
import matplotlib.pyplot as plt
# Suppress runtime warnings (like log(0)) during initial training steps if any
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Activation Functions ---
def sigmoid(z):
  """Sigmoid activation function"""
  # Add small epsilon for numerical stability if z is very negative
  # z = np.clip(z, -500, 500) # Optional clipping
  return 1 / (1 + np.exp(-z))

def relu(z):
  """ReLU activation function"""
  return np.maximum(0, z)

def sigmoid_derivative(z):
  """Derivative of the sigmoid function"""
  s = sigmoid(z)
  return s * (1 - s)

def relu_derivative(z):
  """Derivative of the ReLU function"""
  # Create a copy to avoid modifying the original z indirectly if needed
  deriv = np.array(z, copy=True)
  deriv[deriv <= 0] = 0
  deriv[deriv > 0] = 1
  return deriv

# --- Loss Function ---
def compute_cost(Y_hat, Y):
  """
  Computes the binary cross-entropy cost.

  Arguments:
  Y_hat -- The output of the activation function (predictions), shape (1, number of examples)
  Y -- The true "label" vector, shape (1, number of examples)

  Returns:
  cost -- Cross-entropy cost
  """
  m = Y.shape[1] # Number of examples

  # Add epsilon to prevent log(0) errors
  epsilon = 1e-8
  cost = - (1/m) * np.sum(Y * np.log(Y_hat + epsilon) + (1 - Y) * np.log(1 - Y_hat + epsilon))

  # Makes sure cost is the dimension we expect.
  # E.g., turns [[17]] into 17.
  cost = np.squeeze(cost)
  return cost

# --- Parameter Initialization ---
def initialize_parameters(n_x, n_h, n_y):
  """
  Initializes weights and biases for a 2-layer network.

  Arguments:
  n_x -- size of the input layer
  n_h -- size of the hidden layer
  n_y -- size of the output layer

  Returns:
  parameters -- python dictionary containing parameters:
                  W1 -- weight matrix of shape (n_h, n_x)
                  b1 -- bias vector of shape (n_h, 1)
                  W2 -- weight matrix of shape (n_y, n_h)
                  b2 -- bias vector of shape (n_y, 1)
  """
  np.random.seed(2) # Set seed for reproducibility

  # Xavier initialization can sometimes be better, but simple random * 0.01 works here
  W1 = np.random.randn(n_h, n_x) * 0.1 # Small random values
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.1 # Small random values
  b2 = np.zeros((n_y, 1))

  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
  return parameters

# --- Forward Propagation ---
def forward_propagation(X, parameters):
  """
  Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> SIGMOID

  Arguments:
  X -- input data of size (n_x, number of examples)
  parameters -- python dictionary containing parameters (W1, b1, W2, b2)

  Returns:
  Y_hat -- The output of the last ACTIVATION function (prediction)
  cache -- a dictionary containing "Z1", "A1", "Z2", "A2"
  """
  # Retrieve parameters
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']

  # Implement Forward Propagation to calculate A2 (probabilities)
  Z1 = np.dot(W1, X) + b1 # Weighted sum for hidden layer
  A1 = relu(Z1)         # Activation for hidden layer
  Z2 = np.dot(W2, A1) + b2 # Weighted sum for output layer
  Y_hat = sigmoid(Z2)    # Activation for output layer (prediction)

  assert(Y_hat.shape == (1, X.shape[1])) # Check shape

  cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": Y_hat}
  return Y_hat, cache

# --- Backward Propagation ---
def backward_propagation(parameters, cache, X, Y):
  """
  Implements the backward propagation using the formulas derived.

  Arguments:
  parameters -- python dictionary containing parameters W1, b1, W2, b2
  cache -- dictionary containing "Z1", "A1", "Z2", "A2" from forward prop
  X -- input data of shape (n_x, number of examples)
  Y -- true "label" vector of shape (1, number of examples)

  Returns:
  grads -- A dictionary with the gradients with respect to different parameters
  """
  m = X.shape[1] # Number of examples

  # Retrieve W1 and W2 from parameters
  W1 = parameters['W1']
  W2 = parameters['W2']

  # Retrieve A1 and A2 from cache
  A1 = cache['A1']
  A2 = cache['A2'] # This is Y_hat
  Z1 = cache['Z1']
  # Z2 = cache['Z2'] # Not directly needed if using combined derivative below

  # --- Backward propagation: calculate dW1, db1, dW2, db2. ---
  # Output Layer Gradients (using Sigmoid + Binary Cross-Entropy simplified derivative)
  dZ2 = A2 - Y # Element-wise difference (Shape: (n_y, m))

  # Gradients for W2 and b2
  dW2 = (1/m) * np.dot(dZ2, A1.T) # Shape: (n_y, n_h)
  db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) # Shape: (n_y, 1)

  # Hidden Layer Gradients
  dA1 = np.dot(W2.T, dZ2) # Propagate error back (Shape: (n_h, m))
  dZ1 = dA1 * relu_derivative(Z1) # Element-wise multiplication (Shape: (n_h, m))

  # Gradients for W1 and b1
  dW1 = (1/m) * np.dot(dZ1, X.T) # Shape: (n_h, n_x)
  db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) # Shape: (n_h, 1)

  grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
  return grads

# --- Update Parameters ---
def update_parameters(parameters, grads, learning_rate=1.2):
  """
  Updates parameters using the gradient descent update rule.

  Arguments:
  parameters -- python dictionary containing your parameters
  grads -- python dictionary containing your gradients
  learning_rate -- the learning rate

  Returns:
  parameters -- python dictionary containing your updated parameters
  """
  # Retrieve parameters
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']

  # Retrieve gradients
  dW1 = grads['dW1']
  db1 = grads['db1']
  dW2 = grads['dW2']
  db2 = grads['db2']

  # Update rule for each parameter
  W1 = W1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1
  W2 = W2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2

  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
  return parameters

# --- Build the Neural Network Model ---
def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=0.5, print_cost=False):
  """
  Builds and trains the 2-layer neural network.

  Arguments:
  X -- dataset of shape (n_x, number of examples)
  Y -- labels of shape (1, number of examples)
  n_h -- size of the hidden layer
  num_iterations -- Number of iterations in gradient descent loop
  learning_rate -- Learning rate for gradient descent
  print_cost -- if True, print the cost every 1000 iterations

  Returns:
  parameters -- parameters learnt by the model. They can then be used to predict.
  costs -- list of costs recorded during training
  """
  np.random.seed(3) # Seed for consistency
  n_x = X.shape[0] # Input layer size
  n_y = Y.shape[0] # Output layer size

  # Initialize parameters
  parameters = initialize_parameters(n_x, n_h, n_y)

  costs = [] # To keep track of the cost

  # Gradient descent loop
  for i in range(0, num_iterations):
    # Forward propagation
    Y_hat, cache = forward_propagation(X, parameters)

    # Cost function
    cost = compute_cost(Y_hat, Y)

    # Backpropagation
    grads = backward_propagation(parameters, cache, X, Y)

    # Gradient descent parameter update
    parameters = update_parameters(parameters, grads, learning_rate)

    # Record the cost
    if i % 100 == 0:
        costs.append(cost)

    # Print the cost every 1000 iterations
    if print_cost and i % 1000 == 0:
      print(f"Cost after iteration {i}: {cost}")

  return parameters, costs

# --- Prediction Function ---
def predict(parameters, X):
  """
  Using the learned parameters, predicts a class for each example in X

  Arguments:
  parameters -- python dictionary containing your parameters
  X -- input data of size (n_x, m)

  Returns
  predictions -- vector of predictions (0 or 1) for each example
  """
  # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 threshold.
  Y_hat, cache = forward_propagation(X, parameters)
  predictions = (Y_hat > 0.5).astype(int) # Convert probabilities to 0/1 predictions

  return predictions

# --- Plotting Function ---
def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary learned by the model.

  Arguments:
  model -- a function that takes parameters and X, and returns predictions
  X -- input features, shape (n_x, m)
  y -- true labels, shape (1, m)
  """
  # Set min and max values and give it some padding
  x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
  y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
  h = 0.01 # step size in the mesh

  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Predict the function value for the whole grid
  # Need to transpose the concatenated grid points to match model input (n_x, m)
  Z = model(np.c_[xx.ravel(), yy.ravel()].T)
  Z = Z.reshape(xx.shape) # Reshape predictions to match grid shape

  # Plot the contour and training examples
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
  plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral, edgecolors='k') # Use squeeze on y
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('Decision Boundary')
  plt.show()


# --- Main Execution ---

# 1. Generate XOR Data
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) # Shape (2, 4) -> n_x = 2, m = 4
Y = np.array([[0, 1, 1, 0]]) # Shape (1, 4) -> n_y = 1

# 2. Define Network Structure
n_x = X.shape[0] # Size of input layer
n_h = 4         # Size of hidden layer (can be tuned)
n_y = Y.shape[0] # Size of output layer

# 3. Train the Model
learning_rate = 0.1 # Adjusted learning rate
num_iterations = 20000 # Increased iterations for convergence
parameters, costs = nn_model(X, Y, n_h, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)

# 4. Evaluate the Model (optional)
predictions = predict(parameters, X)
accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / Y.size * 100)
print(f'\nAccuracy on training set: {accuracy:.2f} %')

# 5. Visualize Results

# Plot cost function
plt.figure()
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title(f'Cost Reduction over Training (LR={learning_rate})')
plt.show()

# Plot decision boundary
plt.figure()
# Define a lambda function compatible with plot_decision_boundary
model_predictor = lambda x: predict(parameters, x)
plot_decision_boundary(model_predictor, X, Y)