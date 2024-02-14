import numpy as np

# Define PCA class
class PCA:
    """
    Principal Component Analysis (PCA) implementation using NumPy from scratch.

    Parameters:
    -----------
    n_components : int or None, optional (default=None)
    Number of principal components to retain. If None, all components are retained.

    """

    def __init__(self, n_components = None):
        self.n_components = n_components

    # fit() method
        def fit(self, X):
         """
        Fit the PCA model to the data.

        Parameters:
        -----------
        X : matrix

        Returns:
        --------
        self : returns the instance itself.
        """
        # Calculate covariance matrix
        cov_matrix = np.cov(X.T)

        # Calculate eigenvalues and vectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Order the eigenvectors by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        self.components_ = eigenvectors[:,idx]

        # Keep specified number of components
        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
        
        return self

    # fit_transform() method
    def fit_transform(self, X):
        """
        Fit the PCA model to the data and transform the data.

        Parameters:
        -----------
        X : matrix

        Returns:
        --------
        X_transformed : matrix, shape (n_samples, n_components) -> Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    # transform() method
    def transform(self, X):
        return np.dot(X, self.components_)