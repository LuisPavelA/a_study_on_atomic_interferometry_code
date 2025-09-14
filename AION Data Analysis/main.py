# Compiled from a Kaggle notebook
# h5 written by Dr Charles Baynham

import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
import h5py
from pathlib import Path
import math

file = "/kaggle/input/atomic-interferometry-data-aion/000033761-DifferentialClockInterferometryFrag.h5"

f = h5py.File(file)

images_groundstate = np.array(f["datasets"]["ndscan.rid_33761.points.channel_andor_image_0"])
images_excitedstate = np.array(f["datasets"]["ndscan.rid_33761.points.channel_andor_image_1"])

images_groundstate_bg = np.array(f["datasets"]["ndscan.rid_33761.points.channel_andor_image_2"])
images_excitedstate_bg = np.array(f["datasets"]["ndscan.rid_33761.points.channel_andor_image_3"])

images_groundstate_corrected = images_groundstate - images_groundstate_bg
images_excitedstate_corrected = images_excitedstate - images_excitedstate_bg

applied_phase = np.array(f["datasets"]["ndscan.rid_33761.points.axis_0"])

# Erroneous Data Deletion
images_groundstate_corrected = np.delete(images_groundstate_corrected, 523, 0)
images_excitedstate_corrected = np.delete(images_excitedstate_corrected, 523, 0)

images_groundstate_corrected = np.delete(images_groundstate_corrected, 707, 0)
images_excitedstate_corrected = np.delete(images_excitedstate_corrected, 707, 0)

number_of_images = len(images_groundstate_corrected)

restricted_groundstate_images = []
restricted_excitedstate_images = []

for image_index in range(0, number_of_images):
    restricted_groundstate_images.append(images_groundstate_corrected[image_index].T[100:200, 140:235].T)
    restricted_excitedstate_images.append(images_excitedstate_corrected[image_index].T[100:200, 140:235].T)

restricted_groundstate_images = np.array(restricted_groundstate_images)
restricted_excitedstate_images = np.array(restricted_excitedstate_images)

top_images = []
bottom_images = []

for image_index in range(0, number_of_images):
    top_image = np.concatenate((restricted_groundstate_images[image_index].T[0:50].T, restricted_groundstate_images[image_index].T[50:100].T), axis=1)
    bottom_image = np.concatenate((restricted_excitedstate_images[image_index].T[0:50].T, restricted_excitedstate_images[image_index].T[50:100].T), axis=1)
    
    top_images.append(top_image)
    bottom_images.append(bottom_image)

top_images = np.array(top_images)
bottom_images = np.array(bottom_images)

# Getting the input data in the right format to perform PCA

top_images_dataframe = []
bottom_images_dataframe = []

for image_index in range(0, number_of_images):
    top_image = pd.Series(top_images[image_index].flatten())
    bottom_image = pd.Series(bottom_images[image_index].flatten())
    
    top_images_dataframe.append(top_image)
    bottom_images_dataframe.append(bottom_image)

top_images_dataframe = pd.DataFrame(top_images_dataframe).T
bottom_images_dataframe = pd.DataFrame(bottom_images_dataframe).T

# Performing PCA
pca_transform_top_images = PCA(n_components = 2)
coefficients_top_images = pca_transform_top_images.fit_transform(top_images_dataframe.T)

pca_transform_bottom_images = PCA(n_components = 2)
coefficients_bottom_images = pca_transform_bottom_images.fit_transform(bottom_images_dataframe.T)

# Mean Image
fig, axs = plt.subplots(1, 2)

axs[0].imshow(pca_transform_top_images.mean_.reshape(95, 100))
axs[0].set_title("Ground State")

axs[1].imshow(pca_transform_bottom_images.mean_.reshape(95, 100))
axs[1].set_title("Excited State")

plt.savefig("mean_image.pdf", format="pdf", bbox_inches="tight")

# First Principal Component
fig, axs = plt.subplots(1, 2)

axs[0].imshow(pca_transform_top_images.components_[0].reshape(95, 100))
axs[0].set_title("Ground State")

axs[1].imshow(pca_transform_bottom_images.components_[0].reshape(95, 100))
axs[1].set_title("Excited State")

plt.savefig("first_principal_component_image.pdf", format="pdf", bbox_inches="tight")

# Second Principal Component
fig, axs = plt.subplots(1, 2)

axs[0].imshow(pca_transform_top_images.components_[1].reshape(95, 100))
axs[0].set_title("Ground State")

axs[1].imshow(pca_transform_bottom_images.components_[1].reshape(95, 100))
axs[1].set_title("Excited State")

plt.savefig("second_principal_component_image.pdf", format="pdf", bbox_inches="tight")

# Coefficients Top
w1_top = []
w2_top = []

for set_of_coefficients in coefficients_top_images:
    w1_top.append(set_of_coefficients[0])
    w2_top.append(set_of_coefficients[1])

# Coefficients Bottom
w1_bottom = []
w2_bottom = []

for set_of_coefficients in coefficients_bottom_images:
    w1_bottom.append(set_of_coefficients[0])
    w2_bottom.append(set_of_coefficients[1])

# Coefficient Scattering
fig, axs = plt.subplots(1, 2)

axs[0].scatter(w2_top, w1_top, color = "blue")
axs[0].axis("equal")
axs[0].set_title("Ground State")
axs[0].set_xlabel("$\mathrm{w}_{\mathrm{2}}$")
axs[0].set_ylabel("$\mathrm{w}_{\mathrm{1}}$")

axs[1].scatter(w2_bottom, w1_bottom, color = "blue")
axs[1].axis("equal")
axs[1].set_title("Excited State")
axs[1].set_xlabel("$\mathrm{w}_{\mathrm{2}}$")
axs[1].set_ylabel("$\mathrm{w}_{\mathrm{1}}$")
fig.subplots_adjust(wspace=0.5)

plt.savefig("scattered_coefficients.pdf", format="pdf", bbox_inches="tight")

# Ellipse Fitting
fig, axs = plt.subplots(1, 2)

ellipse_data_top = np.array(list(zip(w2_top, w1_top)))
ellipse_data_bottom = np.array(list(zip(w2_bottom, w1_bottom)))

reg = LsqEllipse().fit(ellipse_data_top)
ellipse_center_top, ellipse_width_top, ellipse_height_top, ellipse_phi_top = reg.as_parameters()
axs[0].axis("equal")
axs[0].plot(w2_top, w1_top, "ro", zorder=1, color="blue")
ellipse = Ellipse(
    xy = ellipse_center_top, width = 2 * ellipse_width_top, height = 2 * ellipse_height_top, angle = np.rad2deg(ellipse_phi_top),
    edgecolor = "r", fc = "None", lw = 2, zorder = 2
)
axs[0].add_patch(ellipse)

axs[0].set_xlabel("$\mathrm{w}_{\mathrm{2}}$")
axs[0].set_ylabel("$\mathrm{w}_{\mathrm{1}}$")

axs[0].set_title("Ground State")

reg = LsqEllipse().fit(ellipse_data_bottom)
ellipse_center_bottom, ellipse_width_bottom, ellipse_height_bottom, ellipse_phi_bottom = reg.as_parameters()
axs[1].axis("equal")
axs[1].plot(w2_bottom, w1_bottom, "ro", zorder=1, color="blue")
ellipse = Ellipse(
    xy = ellipse_center_bottom, width = 2 * ellipse_width_bottom, height = 2 * ellipse_height_bottom, angle = np.rad2deg(ellipse_phi_bottom),
    edgecolor = "r", fc = "None", lw = 2, zorder = 2
)
axs[1].add_patch(ellipse)

axs[1].set_title("Excited State")
axs[1].set_xlabel("$\mathrm{w}_{\mathrm{2}}$")
axs[1].set_ylabel("$\mathrm{w}_{\mathrm{1}}$")

fig.subplots_adjust(wspace=0.5)

plt.savefig("plotted_and_fitted_coefficients.pdf", format="pdf", bbox_inches="tight")

# Coefficient Correction
w1_top_prime = []
w2_top_prime = []

w1_bottom_prime = []
w2_bottom_prime = []

angle_of_rotation_top = ellipse_phi_top
angle_of_rotation_bottom = ellipse_phi_bottom

coefficients_top = coefficients_top_images.copy()
coefficients_bottom = coefficients_top_images.copy()

rotation_matrix_top = np.array([[np.cos(angle_of_rotation_top), np.sin(angle_of_rotation_top)], [-1 * np.sin(angle_of_rotation_top), np.cos(angle_of_rotation_top)]])
scaling_matrix_top = np.array([[1/ellipse_height_top, 0], [0, 1/ellipse_width_top]])

rotation_matrix_bottom = np.array([[np.cos(angle_of_rotation_bottom), np.sin(angle_of_rotation_bottom)], [-1 * np.sin(angle_of_rotation_bottom), np.cos(angle_of_rotation_bottom)]])
scaling_matrix_bottom = np.array([[1/ellipse_height_bottom, 0], [0, 1/ellipse_width_bottom]])

for coefficient_index in range(0, number_of_images):
    coefficient_pair_top = coefficients_top[coefficient_index]
    coefficient_pair_bottom = coefficients_bottom[coefficient_index]

    coefficient_pair_top[0] = coefficient_pair_top[0] - ellipse_center_top[0]
    coefficient_pair_top[1] = -1 * coefficient_pair_top[1] - ellipse_center_top[1]

    coefficient_pair_bottom[0] = coefficient_pair_bottom[0] - ellipse_center_bottom[0]
    coefficient_pair_bottom[1] = -1 * coefficient_pair_bottom[1] - ellipse_center_bottom[1]
    
    resulting_coefficient_pair_top = np.matmul(scaling_matrix_top, np.matmul(rotation_matrix_top, coefficient_pair_top))
    
    resulting_coefficient_pair_bottom = np.matmul(scaling_matrix_top, np.matmul(rotation_matrix_bottom, coefficient_pair_bottom))

    w1_top_prime.append(resulting_coefficient_pair_top[0])
    w2_top_prime.append(resulting_coefficient_pair_top[1])

    w1_bottom_prime.append(resulting_coefficient_pair_bottom[0])
    w2_bottom_prime.append(resulting_coefficient_pair_bottom[1])

# Corrected Coefficients with Fitted Ellipses
ellipse_data_top = np.array(list(zip(w2_top_prime, w1_top_prime)))
ellipse_data_bottom = np.array(list(zip(w2_bottom_prime, w1_bottom_prime)))

fig, axs = plt.subplots(1, 2)

reg = LsqEllipse().fit(ellipse_data_top)
ellipse_center, ellipse_width, ellipse_height, ellipse_phi = reg.as_parameters()
axs[0].plot(w2_top_prime, w1_top_prime, "ro", zorder=1, color = "blue")
ellipse = Ellipse(
    xy = ellipse_center, width = 2 * ellipse_width, height = 2 * ellipse_height, angle = np.rad2deg(ellipse_phi),
    edgecolor = "r", fc = "None", lw = 2, label = "Fit", zorder = 2
)
axs[0].add_patch(ellipse)

axs[0].set_title("Ground State")
axs[0].set_xlabel("$\mathrm{w}_{\mathrm{2}}'$")
axs[0].set_ylabel("$\mathrm{w}_{\mathrm{1}}'$")


reg = LsqEllipse().fit(ellipse_data_bottom)
ellipse_center, ellipse_width, ellipse_height, ellipse_phi = reg.as_parameters()
axs[1].plot(w2_bottom_prime, w1_bottom_prime, "ro", zorder=1, color = "blue")
ellipse = Ellipse(
    xy = ellipse_center, width = 2 * ellipse_width, height = 2 * ellipse_height, angle = np.rad2deg(ellipse_phi),
    edgecolor = "r", fc = "None", lw = 2, label = "Fit", zorder = 2
)
axs[1].add_patch(ellipse)

axs[1].set_title("Excited State")
axs[1].set_xlabel("$\mathrm{w}_{\mathrm{2}}$")
axs[1].set_ylabel("$\mathrm{w}^{\prime}_{\mathrm{1}}$")

fig.subplots_adjust(wspace=0.5)

axs[0].axis("equal")
axs[1].axis("equal")

plt.savefig("corrected_scattered_and_fitted_coefficients.pdf", format="pdf", bbox_inches="tight")

# Differential Phase Shift Extraction Attempt
results = []
for i in range(0, number_of_images):
    if math.atan2(w2_top_prime[i], w1_top_prime[i]) - math.atan2(w2_bottom_prime[i], w1_bottom_prime[i]) < 1:
        results.append(math.atan2(w2_top_prime[i], w1_top_prime[i]) - math.atan2(w2_bottom_prime[i], w1_bottom_prime[i]))

# results_array = (2 * np.pi + np.array(results)) * (np.array(results) < 0) + np.array(results) * (np.array(results) > 0)

plt.scatter(range(0, len(results)), results)
print(np.mean(results))

# Spatial Phase Profile Reconstruction Attempt
rotation_matrix_top = np.array([[np.cos(angle_of_rotation_top), -1 * np.sin(angle_of_rotation_top)], [np.sin(angle_of_rotation_top), np.cos(angle_of_rotation_top)]])
scaling_matrix_top = np.array([[1/ellipse_height_top, 0], [0, 1/ellipse_width_top]])

rotation_matrix_bottom = np.array([[np.cos(angle_of_rotation_bottom), -1 * np.sin(angle_of_rotation_bottom)], [np.sin(angle_of_rotation_bottom), np.cos(angle_of_rotation_bottom)]])
scaling_matrix_bottom = np.array([[1/ellipse_height_bottom, 0], [0, 1/ellipse_width_bottom]])

corrected_top_first_component = list()
corrected_top_second_component = list()

corrected_bottom_first_component = list()
corrected_bottom_second_component = list()

for pixel_index in range(0, len(pca_transform_top_images.components_[0])):
    top_pair = np.matmul(scaling_matrix_top, np.matmul(rotation_matrix_top, [pca_transform_top_images.components_[0][pixel_index], pca_transform_top_images.components_[1][pixel_index]]))
    bottom_pair = np.matmul(scaling_matrix_bottom, np.matmul(rotation_matrix_bottom, [pca_transform_bottom_images.components_[0][pixel_index], pca_transform_bottom_images.components_[1][pixel_index]]))
    corrected_top_first_component.append(top_pair[0])
    corrected_top_second_component.append(top_pair[1])
    
    corrected_bottom_first_component.append(bottom_pair[0])
    corrected_bottom_second_component.append(bottom_pair[1])

reconstructed_spatial_phase_profile_top = list()
reconstructed_spatial_phase_profile_bottom = list()

for pixel_index in range(0, len(corrected_bottom_first_component)):
    reconstructed_spatial_phase_profile_top.append(math.atan2(corrected_top_second_component[pixel_index], corrected_top_first_component[pixel_index]))
    reconstructed_spatial_phase_profile_bottom.append(math.atan2(corrected_bottom_second_component[pixel_index], corrected_bottom_first_component[pixel_index]))

plt.imshow(np.array(reconstructed_spatial_phase_profile_top).reshape(95, 100))
plt.imshow(np.array(reconstructed_spatial_phase_profile_bottom).reshape(95, 100))