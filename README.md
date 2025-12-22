Eigenfaces – PCA for Facial Recognition

Transforming high-dimensional facial data into meaningful patterns.

A professional module that computes eigenfaces using PCA from preprocessed face images, enabling visualization, dimensionality reduction, and downstream facial recognition.

Project Overview

This module is part of a collaborative project to implement facial recognition with PCA (Eigenfaces).

Computes the mean face from training data.

Performs eigen decomposition on the reduced covariance matrix.

Generates eigenfaces, the principal components of the face dataset.

Visualizes top eigenfaces and cumulative variance explained.

Saves outputs for projection, reconstruction, and recognition in Member 4.

Features
Step	Description	Status
Load Preprocessed Data	Load X_centered.npy, mean_face.npy, and L.npy from Member 2	Done
Eigen Decomposition	Compute eigenvalues and eigenvectors of reduced covariance	Done
Compute Eigenfaces	Transform eigenvectors to original space and normalize	Done
Visualization	Display top 10 eigenfaces & plot cumulative variance	Done
Save Outputs	Save eigenfaces.npy and eigenvalues.npy for Member 4	Done
Technical Details

Reduced Covariance Trick: Use L = X_centered * X_centered.T to avoid huge D×D covariance.

Eigenfaces Calculation:
eigenfaces = X_centered.T * eigenvectors
Normalize each column vector.

Visualization: Reshape eigenfaces (10000,) → (100,100) for plotting.

Variance Explained: Each eigenvalue represents the variance captured; cumulative variance plot shows contribution of top components.

Data Used

Inputs: X_centered.npy, mean_face.npy, L.npy (preprocessed by Member 2)

Image size: 100 × 100 pixels

Normalization: Pixel values scaled to [0,1]

Outputs

eigenfaces.npy – Eigenfaces in original image space (10000 × N)

eigenvalues.npy – Corresponding eigenvalues (N,)

Visualizations: Top 10 eigenfaces, cumulative variance plot

Used by Member 4 to:

Project faces into eigenface space

Reconstruct faces from selected components

Perform recognition using Euclidean distance

Tech Stack
Layer	Technology
Language	Python
Libraries	NumPy, Matplotlib, Pandas
Data	Preprocessed CelebA-500 images
Output	.npy files for eigenfaces & eigenvalues
Quick Start (Run Locally)
# Clone repo
git clone <your-repo-url>
cd Member3_Eigenfaces

# Install requirements
pip install -r requirements.txt

# Run notebook or script
jupyter notebook Member3_Eigenfaces.ipynb

References

Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1), 71–86.

Python Libraries: NumPy, Matplotlib
