import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define PCA class
class RPCA():
   
    def __init__(self, n_components=None, solver='svd'):
        """Principal Component Analysis (PCA) implementation using NumPy from scratch.
        
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.

        Parameters:
        -----------
        n_components : int or None, optional (default=None)
            Number of principal components to retain. If None, all components are retained.
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.n_components = n_components
        self.solver = solver
        self.scaler = StandardScaler()  

    # fit() method
    def fit(self, X):
        """
        Fit the PCA model to the data. Calculate the eigenvectors from covariance matrix

        Parameters:
        -----------
        X : matrix. Input data matrix, shape (n_samples, n_features).

        Returns:
        --------
        self : returns the instance itself.
        """
        
        # Data preprocessing
        X = self.scaler.fit_transform(X)

        # Mean centering
        self.mean = np.mean(X, axis=0) # Calculate the mean of each feature
        X = X.copy()
        X -= self.mean

        if self.solver == 'eig':
            cov_matrix = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            idx = eigenvalues.argsort()[::-1]   # Order the eigenvectors by eigenvalues (descending)
            self.components_ = eigenvectors[:, idx]
        
        elif self.solver == 'svd':
            U, S, Vt = np.linalg.svd(X)           # Calculate the SVD of the centered data matrix
            self.components_ = Vt.T               # Extract principal components (eigenvectors) from Vt

        # Keep specified number of components
        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
        
        return self

    # transform() method
    def transform(self, X):
        """
        Project data onto principal components.

        Parameters:
        -----------
        X : matrix
            Input data matrix, shape (n_samples, n_features).

        Returns:
        --------
        X_transformed : matrix
            Transformed data matrix, shape (n_samples, n_components).
        """
        X = X.copy()
        return np.dot(X, self.components_)

    # fit_transform() method
    def fit_transform(self, X):
        """Fit the PCA model to the data and transform the data.

        Parameters:
        -----------
        X : matrix
            Input data matrix, shape (n_samples, n_features).

        Returns:
        --------
        X_transformed : matrix
            Transformed data matrix, shape (n_samples, n_components).
        """
        X = X.copy()
        self.fit(X)
        return self.transform(X)

# https://github.com/rushter/MLAlgorithms/blob/master/mla/pca.py