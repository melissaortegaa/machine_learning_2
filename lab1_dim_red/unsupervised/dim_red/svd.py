import numpy as np

# Define SVD class
class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None

    # fit() method
    def fit(self, X):
        # Calculate la descomposición de valores singulares of X matrix and save components U, S and Vt
        self.U, self.S, self.Vt = np.linalg.svd(X)
        print("Self.U:", self.U)
        print("Self.S:", self.S)
        print("Self.Vt:", self.Vt)

        # Mantener solo el número especificado de componentes principales
        if self.n_components is not None:
            self.U = self.U[:, :self.n_components]
            self.S = self.S[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]

        # Devuelve los componentes U, S y Vt
        return self.U, self.S, self.Vt
        
    # transform() method to transform X original data using Vt components
    def transform(self, X):
        return np.dot(X, self.Vt.T)

    # inverse_transform() in case we need to revert the transformation
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.Vt)