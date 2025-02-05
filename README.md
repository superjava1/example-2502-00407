# Causal Abstraction Learning based on the Semantic Embedding Principle

This repository contains an implementation of a simplified version of **Causal Abstraction (CA) Learning** for linear models. The approach is based on the Semantic Embedding Principle (SEP) and leverages optimization on the Stiefel manifold.

In our implementation, we learn a linear mapping \(V^	op\) that projects detailed (low-level) data onto an aggregated (high-level) space while preserving the essential causal structure. The objective function is inspired by a Kullback-Leibler divergence–like metric and is defined as

\[
f(V) = \operatorname{Tr}\!\Bigl(igl(V^	op \Sigma_\ell Vigr)^{-1}\Sigma_h\Bigr) + \log\det\!\Bigl(V^	op \Sigma_\ell V\Bigr) + \lambda\,\| D \odot V \|_1\,,
\]

subject to \(V\) belonging to the Stiefel manifold \(\mathrm{St}(l, h)\), that is, \(V^	op V = I_h\).

The penalty term \( \lambda\,\| D \odot V \|_1 \) enforces prior structural information provided via a binary matrix \(B\) (with \(D = \mathbf{1} - B\)). This helps guide the solution toward mappings that match our domain knowledge.

## Repository Structure

- **`main.py`**  
  Contains the main code to set up the optimization problem, define the cost function, and solve for the optimal mapping \(V\) using the Pymanopt library.

- **`README.md`**  
  This file, which provides an overview of the repository, instructions, and detailed explanations of the implementation.

- **(Additional files and folders can be added as needed.)**

## Prerequisites

Before running the code, ensure you have Python 3 and the following libraries installed:

- `numpy`
- `pymanopt`

You can install these packages using pip:

```bash
pip install numpy pymanopt
```

## How to Run

1. **Run the Main Script:**

   Execute the main Python script to perform the optimization:

   ```bash
   python main.py
   ```

   The script will:
   - Generate simulated covariance matrices for the low-level (\(\Sigma_\ell\)) and high-level (\(\Sigma_h\)) models.
   - Incorporate prior structural information via a binary matrix \(B\) (and corresponding penalty mask \(D\)).
   - Define the cost function and optimize over the Stiefel manifold.
   - Print the optimal matrix \(V\) and verify that the orthogonality condition is met.
   - Provide an example of projecting a sample vector from the low-level space to the high-level space using \(V^	op\).

https://www.arxiv.org/abs/2502.00407

## Detailed Code Explanation

- **Problem Setup:**  
  The dimensions \(l\) (number of low-level variables) and \(h\) (number of high-level variables) are defined. Random positive definite matrices \(\Sigma_\ell\) and \(\Sigma_h\) are generated to simulate data from low- and high-level models.

- **Prior Structural Information:**  
  A binary matrix \(B\) is simulated to represent available prior knowledge. The penalty mask \(D = \mathbf{1} - B\) is used to penalize entries in the learned mapping \(V\) where the prior indicates no connection.

- **Cost Function:**  
  The cost function computes:
  - The projected covariance matrix \(A = V^	op \Sigma_\ell V\).
  - The trace term \(\operatorname{Tr}\!igl(A^{-1}\Sigma_higr)\) and the log-determinant \(\log\det(A)\).
  - An L1 penalty \( \lambda\,\| D \odot V \|_1 \) that encourages the structure specified by \(B\).
  
  If \(A\) is not invertible or is not positive definite, the function returns infinity to reject that candidate solution.

- **Optimization on the Stiefel Manifold:**  
  The optimization is performed on the Stiefel manifold \( \mathrm{St}(l,h) \) (i.e., the set of matrices \(V\) such that \(V^	op V = I\)). This is achieved using the Pymanopt library with a conjugate gradient solver.

- **Using the Learned Mapping:**  
  Once \(V\) is optimized, the high-level representation of a low-level data sample \(x\) is obtained via the projection \(x_{	ext{high}} = V^	op x\).

