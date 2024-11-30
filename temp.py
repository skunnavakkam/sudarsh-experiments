# now we find a matrix R
# applied to each row (activation)
# such that the activation * R is as close to the corresponding row in gemma_2b_it_activations as possible

# Solve for R in the equation: gemma_2b_activations @ R ≈ gemma_2b_it_activations
# Using the normal equation: R = (A^T A)^(-1) A^T B
# where A is gemma_2b_activations and B is gemma_2b_it_activations

# First compute A^T A
ATA = gemma_2b_activations.T @ gemma_2b_activations

# Then compute A^T B
ATB = gemma_2b_activations.T @ gemma_2b_it_activations

# Solve for R
R = np.linalg.solve(ATA, ATB)

# Compute the transformed activations
transformed_activations = gemma_2b_activations @ R

# Calculate error
error = np.mean((transformed_activations - gemma_2b_it_activations) ** 2)
print(f"Mean squared error: {error}")

# print the frobenius norm of the error
frobenius_norm_error = norm(transformed_activations - gemma_2b_it_activations, "fro")
print(f"Frobenius norm of error: {frobenius_norm_error}")
