# Packages
import numpy as np
import math
import json
import cv2
import os

# Modules
from modules.gaussian import gaussian
from modules.theta_function import theta_function
from modules.spatial_phase_profile import spatial_phase_profile

# Data for all of the images

with open("data.json", "r") as file:
    loaded_data = json.load(file)
    print(loaded_data)

number_of_images = loaded_data["number_of_images"]

environent_type = loaded_data["environment"]["env"]

# if environent_type == "dev":
#     current_image.download_image(f"test_image_{loaded_data["current_testing_image"]}.png")
#     loaded_data["current_testing_image"] = loaded_data["current_testing_image"] + 1
#     with open("data.json", "w") as file:
#         file.write(json.dumps(loaded_data, indent=4))
if environent_type == "bulk_test":

    current_images_test = loaded_data["environment"]["current_images_test"]

    os.mkdir(f"./images/images_test_{current_images_test}")
    loaded_data["environment"]["current_images_test"] = current_images_test + 1
    with open("data.json", "w") as file:
        file.write(json.dumps(loaded_data, indent=4))

class Image:

    def change_pixel(self, x_coordinate, y_coordinate, new_value):
        return 0

    def download_image(self, directory):
        cv2.imwrite(f"./{directory}", self.pixel_array)

    def __init__(self, width, height, index):
        self.width = width
        self.height = height
        self.index = index
        self.pixel_array = np.full((self.height, self.width), 0)
        self.origin_vertical = math.floor(height/2)
        self.origin_horizontal = math.floor(width/2)


for image_index in range(0, number_of_images):

    current_image = Image(loaded_data["image_data"]["image_width"] + 1, loaded_data["image_data"]["image_height"] + 1, image_index)
    for vertical_pixel in range(0, current_image.height):
        for horizontal_pixel in range(0, current_image.width):
            scaling_factor = loaded_data["image_data"]["scaling_factor"]

            x_coordinate = (horizontal_pixel - current_image.origin_horizontal) * scaling_factor
            y_coordinate = (vertical_pixel - current_image.origin_vertical) * scaling_factor

            port_shift_value = loaded_data["image_data"]["port_distance_from_origin"] * loaded_data["image_data"]["scaling_factor"]

            gaussian_intensity = loaded_data["image_data"]["intensity_factor"]

            if (x_coordinate) >= 0:
                theta_function_value = theta_function(current_image.index, loaded_data["number_of_images"])
                spatial_phase_profile_value = spatial_phase_profile(x_coordinate - port_shift_value, y_coordinate)
                phase_function = theta_function_value + spatial_phase_profile_value

                gaussian_value = gaussian(x_coordinate - port_shift_value, y_coordinate, gaussian_intensity)

                current_image.pixel_array[vertical_pixel][horizontal_pixel] = 50 * gaussian_value * (1 + math.cos(phase_function))
            elif (x_coordinate) < 0:
                theta_function_value = theta_function(current_image.index, loaded_data["number_of_images"])
                spatial_phase_profile_value = spatial_phase_profile((x_coordinate + port_shift_value), y_coordinate)
                phase_function = theta_function_value + spatial_phase_profile_value + math.pi

                gaussian_value = gaussian(x_coordinate + port_shift_value, y_coordinate, gaussian_intensity)
                
                current_image.pixel_array[vertical_pixel][horizontal_pixel] = 50 * gaussian_value * (1 + math.cos(phase_function))

    current_image.download_image(f"images/images_test_{current_images_test}/image_{image_index}.png")
    print(f"Image {image_index + 1}/{number_of_images}")

    del current_image

with open("./images/images_test_{current_images_test}/data.json", "x") as file:
    file.write(json.dumps(loaded_data, indent=4))