import imageio
from os import listdir

test_data_input_id = 33

files = listdir(f"./images/images_test_{test_data_input_id}")

images = []

for index in range(0, len(files)):
    images.append(imageio.imread(f"./images/images_test_{test_data_input_id}/image_{index}.png"))

imageio.mimsave(f"./videos/images_test_{test_data_input_id}.mp4", images)