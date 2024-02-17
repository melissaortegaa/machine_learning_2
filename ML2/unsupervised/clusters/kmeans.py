import numpy as np

class KMEANS:
    def __init__(self):
        pass

    # Step 1. Randomly initializing K centroid by picking K samples from X
    def initialize_random_centroids(K, X):
        """Initializes and returns k random centroids"""
        
        m, n = np.shape(X)
        
        # Initializate empty centroids variable with (K, n) shape
        centroids = np.empty((K, n))

        for i in range(K):
            # pick a random data point from X as the centroid
            centroids[i] =  X[np.random.choice(range(m))] 
        return centroids

    # Calculate the distance between two vectors
    def euclidean_distance(self,x1, x2):
        """Calculates and returns the euclidean distance between two vectors x1 and x2"""
        return np.linalg.norm(x1 - x2)

    # Step 2. 
    def closest_centroid(self, x, centroids, K):
        """Finds and returns the index of the closest centroid for a given vector x"""
        distances = np.empty(K)
        for i in range(K):
            distances[i] = self.euclidean_distance(centroids[i], x)
        return np.argmin(distances) # return the index of the lowest distance

    # Step 3. Create clusters
    def create_clusters(centroids, K, X):
        """Returns an array of cluster indices for all the data samples"""
        m, _ = np.shape(X)

        cluster_idx = np.empty(m)
        
        for i in range(m):
            cluster_idx[i] = closest_centroid(X[i], centroids, K)
        return cluster_idx