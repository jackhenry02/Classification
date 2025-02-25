import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_ascent(X, y, beta_init, eta, max_iters=1000, tol=1e-6):
    """Perform gradient ascent and return the number of iterations until convergence."""
    beta = beta_init
    for i in range(max_iters):
        p = sigmoid(X @ beta)  # Predicted probabilities
        gradient = X.T @ (y - p)  # Vectorized gradient
        beta_new = beta + eta * gradient  # Update beta
        
        if np.linalg.norm(beta_new - beta) < tol:  # Check convergence
            return i + 1  # Return number of iterations
        
        beta = beta_new
    
    return max_iters  # Return max_iters if it doesn't converge

# Generate some synthetic data for testing
np.random.seed(42)
n_samples, n_features = 100, 3
X = np.random.randn(n_samples, n_features)
y = (np.random.rand(n_samples) > 0.5).astype(int)  # Random binary labels
beta_init = np.zeros(n_features)

# Test different learning rates
eta_values = np.logspace(-4, 0, 20)  # 20 values from 10^(-4) to 10^0 (1)
iterations = []

# Track the minimum iterations and corresponding eta
min_iters = float('inf')
best_eta = None

for eta in eta_values:
    iters = gradient_ascent(X, y, beta_init, eta)
    iterations.append(iters)
    
    if iters < min_iters:
        min_iters = iters
        best_eta = eta

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(eta_values, iterations, marker='o', linestyle='-')
plt.xscale('log')  # Log scale for better visualization
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Number of Iterations to Converge")
plt.title("Effect of Learning Rate on Convergence Speed")
plt.grid(True)
plt.show()

# Print the best eta and corresponding iterations
print(f"The best eta is {best_eta} with {min_iters} iterations.")
