import numpy as np

class KMEANS:
    def __init__(self):
        pass

    # Step 1. Randomly initializing K centroid by picking K samples from X
    def initialize_random_centroids(self, K, X):
        """Initializes and returns K random centroids"""
        
        m, n = np.shape(X)
        centroids = []                                                  # Inicializar los centroides como una lista vacía   
        centroid_indices = np.random.choice(m, size=K, replace=False)   # Inicializar los índices de los puntos seleccionados como centroides
        
        # Seleccionar los puntos correspondientes a los centroides
        for idx in centroid_indices:
            centroids.append(X[idx])          
        centroids = np.array(centroids)     # Convertir la lista de centroides a un arreglo numpy
        
        return centroids

    # Calculate the distance between two vectors
    def _euclidean_distance(self, x1, x2):
        """Calculates and returns the euclidean distance between two vectors x1 and x2"""
        return np.linalg.norm(x1 - x2)

    # Step 2. Find the closest centroid for each data
    def _closest_centroid(self, x, centroids, K):
        """Finds and returns the index of the closest centroid for a given vector x"""
        distances = np.empty(K)
        
        for i in range(K):
            distances[i] = self._euclidean_distance(centroids[i], x)
        return np.argmin(distances) # return the index of the lowest distance

    # Step 3. Create clusters
    def _create_clusters(self, centroids, K, X):
        """Returns an array of cluster indices for all the data samples"""
        m, _ = np.shape(X)
        cluster_idx = np.empty(m)
        
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], centroids, K)  # Corrected function call
        return cluster_idx
    
    # Step 4. Compute the means of each cluster
    def _compute_means(self, cluster_idx, K, X):
        """Computes and returns the new centroids of the clusters"""
        _, n = np.shape(X)
        centroids = np.empty((K, n))
        for i in range(K):
            points = X[cluster_idx == i]           # gather points for the cluster i
            centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points
        return centroids

    # Implementing fit method 
    def fit(self, K, X, max_iterations=500):
        """Runs the K-means algorithm and computes the final clusters"""
        centroids = self.initialize_random_centroids(K, X)
        print(f"Initial centroids:\n {centroids}\n")

        for _ in range(max_iterations):
            clusters = self._create_clusters(centroids, K, X)
            previous_centroids = centroids
            centroids = self._compute_means(clusters, K, X)
            diff = np.abs(previous_centroids - centroids).sum()

            if diff < 1e-5:  # Consideramos que los centroides han convergido si la diferencia es pequeña
                self.centroids = centroids
                print(f"Final centroids:\n {self.centroids}\n")
                return clusters

        self.centroids = centroids
        return clusters
    
    # Implementing transform method
    def predict(self, X):
        """Predicts the closest cluster for each data point in X"""
        if self.centroids is None:
            raise RuntimeError("Fit the model first before making predictions")
        
        m, _ = np.shape(X)
        cluster_idx = np.empty(m)
        
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], self.centroids, len(self.centroids))
        
        return cluster_idx