import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# Loading the dataset
cat_path_template = "datasets/training/cats/cat.{number}.jpg"
dog_path_template = "datasets/training/dogs/dog.{number}.jpg"
cat_img_paths = [cat_path_template.format(number=4000+n) for n in range(1003)]
dog_img_paths = [dog_path_template.format(number=4000+n) for n in range(1002)]

show_image(cat_img_paths[random.randint(0, 1002)])
show_image(dog_img_paths[random.randint(0, 1001)])




