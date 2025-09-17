import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):
    """
    Implement k-Means clustering algorithm
    
    k-Means clustering is a method for partitioning n points into k clusters,
    aiming to group similar points together and represent each group with its "center" (centroid).
    
    Args:
        points: List of points, where each point is a coordinate tuple
        k: Integer representing the number of clusters to form
        initial_centroids: List of initial centroid points, each point is a coordinate tuple
        max_iterations: Integer representing the maximum number of iterations to perform
    
    Returns:
        List of final centroids, each centroid rounded to 4 decimal places, represented as tuples
    """
    # Convert points and centroids to numpy arrays for easier computation
    points = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)
    
    # Perform k-Means iterations
    for iteration in range(max_iterations):
        # Step 1: Assign each point to the nearest centroid
        # Calculate distances from each point to each centroid
        distances = np.sqrt(((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Assign each point to the closest centroid
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Step 2: Update centroids based on assigned points
        new_centroids = np.zeros_like(centroids)
        
        for i in range(k):
            # Find points assigned to cluster i
            cluster_points = points[cluster_assignments == i]
            
            if len(cluster_points) > 0:
                # Calculate new centroid as the mean of assigned points
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned to this cluster, keep the old centroid
                new_centroids[i] = centroids[i]
        
        # Check for convergence (if centroids haven't changed much)
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
            
        # Update centroids for next iteration
        centroids = new_centroids
    
    # Round centroids to 4 decimal places and convert back to tuples
    final_centroids = [tuple(float(x) for x in np.round(centroid, 4)) for centroid in centroids]
    
    return final_centroids

def main():
    # Read input for points
    points = eval(input())
    # Read input for number of clusters
    k = int(input())
    # Read input for initial centroids
    initial_centroids = eval(input())
    # Read input for maximum iterations
    max_iterations = int(input())
    
    # Perform k-Means clustering
    final_centroids = k_means_clustering(points, k, initial_centroids, max_iterations)
    
    # Print the final centroids
    print(final_centroids)

if __name__ == "__main__":
    main()
