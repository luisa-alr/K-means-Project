## Data Mining - K-Means Implementation from Scratch

## Luisa Rosa - Spring 2024

## Instructions:

- Download all files (2 Python programs and 1 PNG image)
- To see the answers to the questions, run the respective python program
  - Run recolor.py to implement the k-means clustering algorithm and apply it to color a given image
  - Run original_color.py to implement the k-means clustering algorithm and reconstruct the coloring of a given image (k determines the number of clusters/colors of the image)

---

## Question 1:

Implement k-means clustering algorithm and apply it to color a given image (image.png).

1. Step 1: First, load the image, which will give you a handle (i.e., img) of a (244, 198, 3) numpy.ndarray. The first two dimensions represent the height andwidth of the image. The last dimension represents the 3 color channels (RGB) for eachpixel of the image.

2. Step 2: Next implement the k-means algorithm to partition the 244×198 pixels into k clustersbased on their RGB values and the Euclidean distance measure. The k-means max iteration is 50. Run your experimentwith k = 2, 3, 6, 10 with the following given starting centroids:
   k = 2: (0, 0, 0), (0.1, 0.1, 0.1)
   k = 3: (0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2)
   k = 6: (0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5,0.5)
   k = 10: (0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5,0.5), (0.6, 0.6, 0.6), (0.7, 0.7, 0.7), (0.8, 0.8, 0.8), (0.9, 0.9, 0.9)

3. Step 3: For each k = 2, 3, 6, 10, report the final SSE and re-color the pixels in each clusterusing the following color scheme:
   Cluster 1. SpringGreen: (60, 179, 113)
   Cluster 2. DeepSkyBlue: (0, 191, 255)
   Cluster 3. Yellow: (255, 255, 0)
   Cluster 4. Red: (255, 0, 0)
   Cluster 5. Black: (0, 0, 0)
   Cluster 6. DarkGray: (169, 169, 169)
   Cluster 7. DarkOrange: (255, 140, 0)
   Cluster 8. Purple: (128, 0, 128)
   Cluster 9. Pink: (255, 192, 203)
   Cluster 10. White: (255, 255, 255)

4. Step 4: Calculate and record the SSE values and colored images for each k. You can find the SSE values on 'sse_values.csv'.

---

## Question 2:

Consider the following dataset: { 0, 4, 5, 20, 25, 39, 43, 44 }

- a) Build a dendrogram for this dataset using the single-link, bottom-up approach.Show your work.
- b) Suppose we want the two top-level clusters. List the data points in each cluster

* Answers included in the written answers PDF

---

## Question 3:

Given two clusters C1 = {(1, 1), (2, 2), (3, 3)} and C2 = {(5, 2), (6, 2), (7, 2), (8, 2), (9, 2)} compute the values in (a) - (f). Use the definition for scattering criteria presented inclass. Note that tr in the scattering criterion is referring to the trace of the matrix.

- a) The mean vectors m1 and m2
- b) The total mean vector m
- c) The scatter matrices S1 and S2
- d) The within-cluster scatter matrix SW
- e) The between-cluster scatter matrix SB
- f) The scatter criterion tr(SB)/tr(SW)

* Answers included in the written answers PDF

---

## Question 4:

A Naive Bayes classifier gives the predicted probability of each data pointbelonging to the positive class, sorted in a descending order:

Instance # | True Class Label | Predicted Probability of Positive Class
1 | P | 0.95
2 | N | 0.85
3 | P | 0.78
4 | P | 0.66
5 | N | 0.60
6 | P | 0.55
7 | N | 0.43
8 | N | 0.42
9 | N | 0.41
10 | P | 0.4

Suppose we use 0.5 as the threshold to assign the predicted class label to each datapoint, i.e., if the predicted probability ≥ 0.5, the data point is assigned to positive class; otherwise, it is assigned to negative class. Calculate the Confusion Matrix, Accuracy, Precision, Recall, F1 Score and Specificity of the classifier.

* Answers included in the written answers PDF

---