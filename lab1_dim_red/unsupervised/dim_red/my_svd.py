import numpy as np
from scipy.linalg import svd

# Define Singular values decomposition (SVD) class
class SVD:
    """
    Singular Value Decomposition (SVD) implementation using NumPy from scratch.
    
    SVD is a factorization of complex matrix that generalizes the eigendecomposition
    of a square normal matrix to any matrix.

    Parameters:
    -----------
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, all components are retained.

    """
    def __init__(self, n_components = None):
        self.n_components = n_components

    # fit() method
    def fit(self, X):
        """
        Fit the SVD model to the data.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features) -> Training data.

        Returns:
        --------
        U : array, shape (n_samples, n_components) or None      -> Left singular vectors
        S : array, shape (n_components,) or None                -> Singular values
        Vt : array, shape (n_components, n_features) or None    -> Right singular vectors
        """

        # Calculate la descomposición de valores singulares of X matrix and save arrays U, S and Vt
        self.U, self.S, self.Vt = np.linalg.svd(X)
        
        # Keep only the specified number of components
        if self.n_components is not None:
            self.U = self.U[:, :self.n_components]
            self.S = self.S[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]


        return self.U, self.S, self.Vt
        
    # transform() method to transform X original data using Vt components
    def transform(self, X):
        """
        Transform X original data using Vt components.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features) -> Data to be transformed.

        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components) -> Transformed data.
        """
        if n_components is None:
            n_components = self.n_components

        return np.dot(X, self.Vt.T)     # where is S used? is missing the @ np.diag(S)

    # inverse_transform() in case we need to revert the transformation
    def inverse_transform(self, X_transformed):
        """
        Revert the transformation of X_transformed.

        Parameters:
        -----------
        X_transformed : array-like, shape (n_samples, n_components) -> Transformed data.

        Returns:
        --------
        X : array-like, shape (n_samples, n_features) -> Original data.
        """
        return np.dot(X_transformed, self.U.T)