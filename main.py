import numpy as np
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import ConjugateGradient

# =============================================================================
# Set up the problem dimensions and random seed for reproducibility
# =============================================================================

np.random.seed(42)  # Fix the seed so that results are reproducible

# Define the dimensions:
#   l: number of variables in the low-level model (detailed features)
#   h: number of variables in the high-level model (aggregated indicators)
l = 10   # e.g., detailed features from transactional data
h = 3    # e.g., aggregated indicators such as volatility, liquidity, volume

# =============================================================================
# Generate covariance matrices for low-level and high-level models
# =============================================================================

# Create a random matrix and generate a positive definite covariance matrix for the low-level model.
# We use A_random^T @ A_random to ensure positive definiteness and add a small constant to the diagonal for stability.
A_random = np.random.randn(l, l)
Sigma_l = A_random.T @ A_random + 0.1 * np.eye(l)

# Similarly, generate a random positive definite covariance matrix for the high-level model.
B_random = np.random.randn(h, h)
Sigma_h = B_random.T @ B_random + 0.1 * np.eye(h)

# =============================================================================
# Incorporate prior structural information as a binary matrix
# =============================================================================

# Assume we have some prior information regarding which elements of the mapping V should be active.
# We simulate this as a binary matrix B_prior of shape (l, h). In practice, this information could come from domain knowledge.
B_prior = np.random.choice([0, 1], size=(l, h))

# Create a penalty mask D which is equal to 1 where B_prior is 0 and 0 where B_prior is 1.
# This means we will penalize entries in V where prior information indicates that there should be no connection.
D = np.ones((l, h)) - B_prior

# Set the regularization parameter lambda_reg for the penalty term.
lambda_reg = 1.0

# =============================================================================
# Define the cost function to be minimized
# =============================================================================

def cost(V):
    """
    Computes the objective function for a given matrix V.
    
    Parameters:
      V (ndarray): A matrix of shape (l, h) that lies on the Stiefel manifold (i.e., V.T @ V = I).
      
    The cost function is defined as:
      f(V) = Tr[(V^T Sigma_l V)^{-1} Sigma_h] + log(det(V^T Sigma_l V)) 
             + lambda_reg * || D ⊙ V ||_1,
    where:
      - The first term (trace term) measures the alignment between the projected low-level covariance and the high-level covariance.
      - The second term (log-det term) ensures proper scaling and numerical conditioning.
      - The third term is an L1 penalty that enforces our prior knowledge on the structure.
    
    Returns:
      float: The computed cost.
    """
    # Compute the projected covariance matrix A = V^T Sigma_l V, which is of shape (h, h)
    A = V.T @ Sigma_l @ V
    
    # Attempt to compute the inverse of A. If A is not invertible, return infinity as a penalty.
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf

    # Compute the log-determinant of A using slogdet for improved numerical stability.
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        # If the determinant is not positive, then A is not positive definite.
        # We penalize this situation by returning infinity.
        return np.inf

    # Compute the trace term: trace((V^T Sigma_l V)^{-1} Sigma_h)
    trace_term = np.trace(invA @ Sigma_h)
    
    # Sum the two terms (trace and log-det) to form the main part of the objective.
    f_val = trace_term + logdet
    
    # Compute the L1 penalty term.
    # The operator D * V represents the element-wise product (Hadamard product)
    # and np.sum(np.abs(...)) computes the L1 norm (sum of absolute values) of the penalized entries.
    penalty = lambda_reg * np.sum(np.abs(D * V))
    
    # Return the total cost.
    return f_val + penalty

# =============================================================================
# Define the manifold on which to optimize: the Stiefel manifold St(l, h)
# =============================================================================

# The Stiefel manifold is the set of matrices V in R^(l x h) such that V.T @ V = I_h.
manifold = Stiefel(l, h)

# =============================================================================
# Set up and solve the optimization problem using Pymanopt
# =============================================================================

# Create a Pymanopt problem instance with the specified manifold and cost function.
problem = Problem(manifold=manifold, cost=cost, verbosity=2)

# Choose a solver. Here we use the Conjugate Gradient solver that is adapted for optimization over manifolds.
solver = ConjugateGradient()

# Solve the problem. This will search for the optimal matrix V that minimizes our cost function
# while ensuring that V remains on the Stiefel manifold.
V_opt = solver.solve(problem)

# =============================================================================
# Display and validate the results
# =============================================================================

print("Optimal matrix V found:")
print(V_opt)

# Check the orthogonality of the columns of V_opt.
# Compute the error: ||V_opt.T @ V_opt - I_h||_F, which should be very close to zero.
ortho_error = np.linalg.norm(V_opt.T @ V_opt - np.eye(h))
print(f"\nOrthogonality error (should be near 0): {ortho_error:.2e}")

# =============================================================================
# Using the learned mapping for causal abstraction
# =============================================================================

# The learned abstraction is given by the transpose of V_opt.
# For example, suppose we have a sample data vector x from the low-level space (dimension l).
# Then, the corresponding high-level representation is computed as:
#
#    x_high = V_opt.T @ x
#
# This projects the detailed low-level data onto the lower-dimensional high-level space,
# preserving the causal structure according to the semantic embedding principle.
#
# Here is an example:
x_sample = np.random.randn(l)  # a random sample from the low-level space
x_high = V_opt.T @ x_sample     # project to high-level space

print("\nExample sample from the low-level space:")
print(x_sample)
print("\nProjected sample in the high-level space:")
print(x_high)
