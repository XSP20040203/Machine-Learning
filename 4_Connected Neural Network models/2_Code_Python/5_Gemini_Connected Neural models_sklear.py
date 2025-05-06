import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs, make_moons # Added make_moons as another option
from sklearn.model_selection import train_test_split # Good practice, though not strictly required for demo
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler # Often crucial for MLP performance

# --- Plotting Function (Adapted for Scikit-learn) ---
def plot_decision_boundary_sklearn(model, X, y, title="Decision Boundary"):
  """
  Plots the decision boundary learned by a scikit-learn model.

  Arguments:
  model -- A trained scikit-learn classifier with a .predict() method.
  X -- Input features used for training/plotting, shape (n_samples, n_features).
       NOTE: Scikit-learn convention!
  y -- True labels, shape (n_samples,).
  title -- Title for the plot.
  """
  # Set min and max values and give it some padding
  # Assumes X has 2 features for visualization
  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
  h = 0.02 # step size in the mesh

  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Predict the function value for the whole grid
  # The model expects input shape (n_samples, n_features)
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape) # Reshape predictions to match grid shape

  # Plot the contour and training examples
  plt.figure(figsize=(8, 6))
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
  # Scatter plot expects y to be 1D for coloring
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title(title)
  plt.show()

# --- 1. Binary Classification (XOR Problem) ---
print("--- Binary Classification (XOR) ---")

# Generate XOR Data (Scikit-learn convention: samples, features)
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]) # Shape (4, 2)
y_xor = np.array([0, 1, 1, 0])      # Shape (4,)

# Scale data (important for MLP)
scaler_xor = StandardScaler()
X_xor_scaled = scaler_xor.fit_transform(X_xor) # Usually fit on train, transform train/test

# Instantiate MLPClassifier
# Parameters similar to our manual setup where possible
mlp_xor = MLPClassifier(
    hidden_layer_sizes=(4,),    # Single hidden layer with 4 neurons
    activation='relu',          # ReLU activation for hidden layer
    solver='adam',              # Optimizer (adam often works well)
    alpha=0.0001,               # L2 regularization (small value)
    batch_size='auto',
    learning_rate_init=0.01,    # Initial learning rate
    max_iter=3000,              # Maximum training iterations (epochs)
    shuffle=True,
    random_state=42,            # For reproducibility
    tol=1e-4,                   # Tolerance for optimization stopping
    verbose=False,              # Set to True to see training progress
    early_stopping=False        # Can be useful on larger datasets
)

# Train the model
print("Training XOR classifier...")
mlp_xor.fit(X_xor_scaled, y_xor)
print("Training complete.")

# Predict
predictions_xor = mlp_xor.predict(X_xor_scaled)

# Evaluate
accuracy_xor = accuracy_score(y_xor, predictions_xor)
print(f"\nAccuracy on XOR data: {accuracy_xor * 100:.2f} %")
print("Classification Report (XOR):")
# target_names optional but good practice
print(classification_report(y_xor, predictions_xor, target_names=['Class 0', 'Class 1']))

# Visualize Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(mlp_xor.loss_curve_)
plt.title("Loss Curve during Training (XOR)")
plt.xlabel("Iterations")
plt.ylabel("Loss (Cross-Entropy)")
plt.grid(True)
plt.show()

# Visualize Decision Boundary
# Use the *original scaled* data for plotting boundary relative to training points
plot_decision_boundary_sklearn(mlp_xor, X_xor_scaled, y_xor, title="Decision Boundary (XOR) - Scaled Data")


print("\n" + "="*30 + "\n")


# --- 2. Multi-Class Classification (Blobs) ---
print("--- Multi-Class Classification (Blobs) ---")

# Generate Data
N_SAMPLES = 1000
N_CLASSES = 5
X_multi, y_multi = make_blobs(n_samples=N_SAMPLES, centers=N_CLASSES, n_features=2,
                              random_state=42, cluster_std=1.2)
# X_multi shape: (400, 2), y_multi shape: (400,) - Already in sklearn format

# Scale data
scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

# Instantiate MLPClassifier for multi-class
# Output layer activation (Softmax) is inferred automatically for multi-class
mlp_multi = MLPClassifier(
    hidden_layer_sizes=(10,),    # Single hidden layer with 10 neurons
    activation='relu',
    solver='adam',
    alpha=0.001,                # Regularization might be more important here
    learning_rate_init=0.001,   # Often start smaller for adam
    max_iter=3000,
    random_state=42,
    verbose=False               # Set to True to monitor convergence
)

# Train the model
print("Training multi-class classifier...")
mlp_multi.fit(X_multi_scaled, y_multi)
print("Training complete.")

# Predict
predictions_multi = mlp_multi.predict(X_multi_scaled)

# Evaluate
accuracy_multi = accuracy_score(y_multi, predictions_multi)
print(f"\nAccuracy on Blobs data: {accuracy_multi * 100:.2f} %")
print("Classification Report (Blobs):")
print(classification_report(y_multi, predictions_multi)) # Auto-detects class names

# Visualize Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(mlp_multi.loss_curve_)
plt.title("Loss Curve during Training (Blobs)")
plt.xlabel("Iterations")
plt.ylabel("Loss (Cross-Entropy)")
plt.grid(True)
plt.show()

# Visualize Decision Boundary
plot_decision_boundary_sklearn(mlp_multi, X_multi_scaled, y_multi, title="Decision Boundary (Blobs) - Scaled Data")