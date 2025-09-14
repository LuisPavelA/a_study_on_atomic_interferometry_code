# Compiled from a Kaggle notebook

import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

input_data_directory = "/kaggle/input/360-image-synthetic-data-linear-phase-pattern"

number_of_files = len(os.listdir(input_data_directory)) 

# Getting the input data in the right format to perform PCA

input_data_array = list()

for image_index in range(0, number_of_files):
    test_image = pd.Series(cv2.imread(f"{input_data_directory}/image_{image_index}.png", cv2.IMREAD_GRAYSCALE).flatten())
    input_data_array.append(test_image)

input_data = pd.DataFrame(input_data_array).T

fig, axes = plt.subplots(int(number_of_files/10), 10, figsize=(12,12), subplot_kw={"xticks":[], "yticks":[]},
 gridspec_kw=dict(hspace=0.01, wspace=0.01))

for i, ax in enumerate(axes.flat):
    ax.imshow(input_data[i].values.reshape(481, 855), cmap="gray")

plt.savefig("linear_dataset_visualization.png")

# PCA Application

pca_transform = PCA(n_components=2)
coefficients = pca_transform.fit_transform(input_data.T)

# First Principal Component

first_principal_component = pca_transform.components_[0].reshape(481, 855)
plt.imshow(first_principal_component, cmap="gray")

plt.savefig("first_principal_component.png")

# Second Principal Component

second_principal_component = pca_transform.components_[1].reshape(481, 855)
plt.imshow(second_principal_component, cmap="gray")

plt.savefig("second_principal_component.png")

# Mean Image

mean_image = pca_transform.mean_.reshape(481, 855)
plt.imshow(mean_image, cmap="gray")

plt.savefig("mean_image.png")

# Formatting the coefficients
w1 = []
w2 = []

for set_of_coefficients in coefficients:
    w1.append(set_of_coefficients[0])
    w2.append(set_of_coefficients[1])

# w1 against w1 Scatter
plt.figure(figsize=(10,10))
plt.scatter(w2, w1)

plt.xlabel("w_2")
plt.ylabel("w_1")

plt.savefig("w1_against_w2.png")

# w1 Scatter
plt.scatter(range(0, 360), w1)

plt.xlabel("Range")
plt.ylabel("w_1")

plt.savefig("w1_scatter.png")

# w2 Scatter
plt.scatter(range(0, 360), w2)

plt.xlabel("Range")
plt.ylabel("w_2")

plt.savefig("w2_scatter.png")