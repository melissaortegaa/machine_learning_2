import numpy as np

class KMedoids:
    def __init__(self):
        pass

    def initialize_random_medoids(self, K, X):
        """Inicializa y devuelve K medoides aleatorios."""
        m, _ = np.shape(X)
        medoids_indices = np.random.choice(m, size=K, replace=False)
        medoids = X[medoids_indices]
        return medoids

    def _manhattan_distance(self, x1, x2):
        """Calcula y devuelve la distancia de Manhattan entre dos vectores."""
        return np.sum(np.abs(x1 - x2))

    def _closest_medoid(self, x, medoids):
        """Encuentra y devuelve el índice del medoide más cercano para un vector dado x."""
        distances = [self._manhattan_distance(x, medoid) for medoid in medoids]
        return np.argmin(distances)

    def _create_clusters(self, medoids, X):
        """Devuelve una matriz de índices de cluster para todas las muestras de datos."""
        m, _ = X.shape
        cluster_idx = np.empty(m)
        for i in range(m):
            cluster_idx[i] = self._closest_medoid(X[i], medoids)
        return cluster_idx

    def _compute_medoids(self, clusters, X):
        """Calcula y devuelve los nuevos medoides."""
        medoids = np.copy(X)  # Usamos X como base para los medoids
        unique_clusters = np.unique(clusters)
        for k in unique_clusters:
            cluster_points = X[clusters == k]
            min_cost = np.inf
            best_medoid_idx = None
            for i, point in enumerate(cluster_points):
                cost = np.sum([self._manhattan_distance(point, q) for q in cluster_points])
                if cost < min_cost:
                    min_cost = cost
                    best_medoid_idx = i
            medoids[clusters == k] = cluster_points[best_medoid_idx]
        return medoids



    def fit(self, K, X, max_iterations=500):
        """Ejecuta el algoritmo K-medoids y calcula los clusters finales."""
        medoids = self.initialize_random_medoids(K, X)
        print(f"Medoids iniciales:\n{medoids}\n")
        
        for _ in range(max_iterations):
            clusters = self._create_clusters(medoids, X)
            previous_medoids = medoids
            medoids = self._compute_medoids(clusters, X)
            if np.array_equal(previous_medoids, medoids):
                return clusters
        
        print(f"Medoids finales:\n{medoids}")
        return clusters
