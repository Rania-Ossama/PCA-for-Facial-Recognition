Overview

This module implements Principal Component Analysis (PCA) for facial recognition. Its main goal is to compute eigenfaces, which are the principal components of a high-dimensional face dataset.
Eigenfaces capture the most significant patterns of variation in facial images and form a lower-dimensional subspace for representing faces.

Responsibilities / Tasks
1. Load Preprocessed Data

Load the mean-centered training data (X_centered.npy) prepared by Member 2.

Load the mean face (mean_face.npy) and reduced covariance matrix (L.npy).

2. Eigen Decomposition

Perform eigenvalue decomposition on the reduced covariance matrix to obtain principal components.

Sort eigenvalues and eigenvectors in descending order of explained variance.

3. Compute Eigenfaces

Transform the eigenvectors from the reduced covariance space to the original image space.

Normalize the resulting eigenfaces.

4. Visualization

Display the top 10 eigenfaces as grayscale images to illustrate dominant facial patterns.

Plot cumulative variance explained by principal components.

5. Save Outputs for Downstream Tasks

Save the eigenfaces (eigenfaces.npy) and eigenvalues (eigenvalues.npy) for use by Member 4 (projection, reconstruction, and recognition).

Data Used

Inputs from Member 2:

X_centered.npy: Mean-centered training images (shape: N × D)

mean_face.npy: Mean face vector (shape: D,)

L.npy: Reduced covariance matrix (shape: N × N)

Assumptions:

Each image is 100 × 100 pixels.

Data has been preprocessed, flattened, and normalized to [0,1].

Outputs

eigenfaces.npy: Eigenfaces in the original image space (shape: D × N)

eigenvalues.npy: Corresponding eigenvalues (shape: N,)

Visual outputs:

Top 10 eigenfaces

Cumulative variance explained plot

Used by Member 4 to:

Project new faces into eigenface space

Reconstruct faces

Perform recognition based on Euclidean distances

Technical Details
Eigenfaces Calculation

Use reduced covariance trick:
L = X_centered * X_centered.T (N × N) to avoid computing huge D × D covariance.

Perform eigen decomposition: np.linalg.eigh(L) → eigenvectors and eigenvalues.

Transform eigenvectors to original space:
eigenfaces = X_centered.T * eigenvectors

Normalize each eigenface:
eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

Visualization

Reshape each eigenface from 10,000-dimensional vector → 100 × 100 pixels

Display grayscale images using matplotlib

Variance Explained

Each eigenvalue corresponds to the variance captured by its eigenface

Cumulative variance plot shows how many eigenfaces are needed to capture most of the dataset’s variance

References

Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1), 71–86.

Python libraries: numpy, matplotlib
