import numpy as np

class SVD:
    """
    Singular Value Decomposition (SVD) implementation using NumPy from scratch.
    
    SVD is a factorization of a complex matrix that generalizes the eigendecomposition
    of a square normal matrix to any matrix.

    Parameters:
    -----------
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, all components are retained.

    """
    def __init__(self, n_components=None):
        self.n_components = n_components

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

        # Calculate the singular value decomposition of X matrix and save arrays U, S, and Vt
        U, S, Vt = np.linalg.svd(X)

        # Keep only the specified number of components
        if self.n_components is not None:
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            Vt = Vt[:self.n_components, :]

        self.U, self.S, self.Vt = U, S, Vt

        return self.U, self.S, self.Vt

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
        if self.n_components is None:
            self.n_components = self.S.shape[0]

        # Scale the transformed data using the singular values
        X_transformed = np.dot(X, self.Vt[:self.n_components].T) @ np.diag(self.S)

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
