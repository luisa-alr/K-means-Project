import pandas as pd
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import os


# Calculate the Euclidean distance between two vectors
def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))


# Initialize centroids with given initial values
def initialize_centroids(k, initial_centroids):
    return np.array(initial_centroids[:k])


# Assign data points to clusters based on nearest centroids
def assign_clusters(data, centroids):
    distances = np.zeros((len(data), len(centroids)))
    for i, point in enumerate(data):
        distances[i] = euclidean_dist(point, centroids)
    return np.argmin(distances, axis=1)


# Recolor pixels based on cluster assignments
def recolor_pixels(cluster_assignments, data):
    color_scheme = {
        0: (60, 179, 113),  # SpringGreen
        1: (0, 191, 255),  # DeepSkyBlue
        2: (255, 255, 0),  # Yellow
        3: (255, 0, 0),  # Red
        4: (0, 0, 0),  # Black
        5: (169, 169, 169),  # DarkGray
        6: (255, 140, 0),  # DarkOrange
        7: (128, 0, 128),  # Purple
        8: (255, 192, 203),  # Pink
        9: (255, 255, 255),  # White
    }
    recolored_pixels = np.zeros((data.shape[0], 3))
    for cluster, color in color_scheme.items():
        recolored_pixels[cluster_assignments == cluster] = color
    return recolored_pixels.astype(np.uint8)


# Calculate sum of squared errors (SSE)
def calculate_sse(data, centroids, cluster_assignments):
    sse = 0
    for cluster, centroid in enumerate(centroids):
        cluster_data = data[cluster_assignments == cluster]
        if len(cluster_data) > 0:
            sse += np.sum((cluster_data - centroid) ** 2)
    return sse


def update_centroids(data, cluster_assignments, k, centroids):
    new_centroids = np.zeros((k, data.shape[1]))
    for cluster in range(k):
        cluster_data = data[cluster_assignments == cluster]
        if len(cluster_data) > 0:
            new_centroids[cluster] = np.mean(cluster_data, axis=0)
        else:
            new_centroids[cluster] = centroids[cluster]
    return new_centroids


# Perform k-means clustering
def kmeans(data, k, initial_centroids, max_iterations=50):
    centroids = initialize_centroids(k, initial_centroids)
    for _ in range(max_iterations):
        prev_centroids = centroids.copy()
        cluster_assignments = assign_clusters(data, centroids)
        centroids = update_centroids(
            data, cluster_assignments, k, centroids
        )  # Pass centroids here
        if np.array_equal(centroids, prev_centroids):
            break
    sse = calculate_sse(data, centroids, cluster_assignments)
    return centroids, cluster_assignments, sse


# Load image
img = skimage.io.imread("image.png")
skimage.io.imshow(img)
plt.show()

# Sample pixel data (replace with your actual pixel data)
pixel_data = img.reshape(
    -1, img.shape[-1]
)  # Reshape to a 2D array where each row represents a pixel (RGB values)

# Experiment with different values of k and initial centroids
k_values = [2, 3, 6, 10]
initial_centroids = {
    2: [(0, 0, 0), (0.1, 0.1, 0.1)],
    3: [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2)],
    6: [
        (0, 0, 0),
        (0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2),
        (0.3, 0.3, 0.3),
        (0.4, 0.4, 0.4),
        (0.5, 0.5, 0.5),
    ],
    10: [
        (0, 0, 0),
        (0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2),
        (0.3, 0.3, 0.3),
        (0.4, 0.4, 0.4),
        (0.5, 0.5, 0.5),
        (0.6, 0.6, 0.6),
        (0.7, 0.7, 0.7),
        (0.8, 0.8, 0.8),
        (0.9, 0.9, 0.9),
    ],
}

# Create a directory to save images if it doesn't exist
if not os.path.exists("output_images"):
    os.makedirs("output_images")

sse_values = []

# Sample pixel data (replace with your actual pixel data)
pixel_data_normalized = pixel_data / 255.0  # Normalize pixel data

# Run k-means for each value of k
for k in k_values:
    print(f"Running k-means with k={k}...")
    centroids, cluster_assignments, sse = kmeans(pixel_data_normalized, k, initial_centroids[k])
    sse_values.append(sse)
    print("Final SSE:", sse)

    # Save the colored image
    recolored_pixels = recolor_pixels(cluster_assignments, pixel_data_normalized)
    recolored_image = recolored_pixels.reshape(244, 198, 3)
    plt.imshow(recolored_image)
    plt.title(f"Recolored Image with k={k}")
    plt.axis("off")
    plt.savefig(f"output_images/recolored_image_k_{k}.png")
    plt.show()

# Save SSE values to a CSV file
sse_df = pd.DataFrame({"k": k_values, "SSE": sse_values})
sse_df.to_csv("sse_values.csv", index=False)

