import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_random_image_path(list_obj: list, max_num: int):
    return list_obj[random.randint(0, max_num)]


def show_image(image_path: str):
    # read the image using OpenCV
    img = cv2.imread(image_path)

    # convert the color from BGR to RGB for displaying using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display the image using matplotlib
    plt.imshow(img)
    plt.title('Cat Image')
    plt.axis('off')
    plt.show()


def show_grayscale_image(image_path: str):
    # Loading the image
    img = cv2.imread(image_path)

    # Converting the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Displaying the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(gray_img, cmap='gray')
    axs[1].set_title('Grayscale Image')
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

# Loading the dataset
cat_path_template = "datasets/training/cats/cat.{number}.jpg"
dog_path_template = "datasets/training/dogs/dog.{number}.jpg"
cat_img_paths = [cat_path_template.format(number=4000+n) for n in range(1003)]
dog_img_paths = [dog_path_template.format(number=4000+n) for n in range(1002)]

# show_image(cat_img_paths[random.randint(0, 1002)])
# show_image(dog_img_paths[random.randint(0, 1001)])

show_grayscale_image(get_random_image_path(cat_img_paths, 1002))
show_grayscale_image(get_random_image_path(dog_img_paths, 1001))


