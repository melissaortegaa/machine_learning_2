import numpy as np

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

def euclidean_distance(x1, x2):
    """Calculates and returns the euclidean distance between two vectors x1 and x2"""
    return np.linalg.norm(x1 - x2)

# Step 2. 
def closest_centroid(x, centroids, K):
    """Finds and returns the index of the closest centroid for a given vector x"""
    distances = np.empty(K)
    
    for i in range(K):
        distances[i] = euclidean_distance(centroids[i], x)
    return np.argmin(distances) # return the index of the lowest distance