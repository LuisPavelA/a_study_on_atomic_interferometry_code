# Compiled from the Notebook

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import random

x, y = np.random.multivariate_normal([0, 0], [[5, 4], [4, 10]], 750).T
random_data = []

for data_element_index in range(0, len(x)):
    random_data.append([x[data_element_index], y[data_element_index]])

input_data_frame = pd.DataFrame(random_data)

pca_transform = PCA(n_components = 2)
coefficients = pca_transform.fit_transform(input_data_frame)

print(pca_transform.components_[0])
print(pca_transform.components_[1])
print(pca_transform.mean_)

plt.scatter(x, y, color="#003dff")

plt.quiver(pca_transform.mean_[1], pca_transform.mean_[0], pca_transform.components_[0][1], pca_transform.components_[0][0], angles='xy', scale_units='xy', scale=0.25, width=0.012, color="#fe0000")
plt.quiver(pca_transform.mean_[1], pca_transform.mean_[0], pca_transform.components_[1][1], pca_transform.components_[1][0], angles='xy', scale_units='xy', scale=0.25, width=0.012, color="#fe0000")

plt.scatter(pca_transform.mean_[1], pca_transform.mean_[0], )

plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)

plt.savefig("PCA_Showcase.pdf", format="pdf", bbox_inches="tight")