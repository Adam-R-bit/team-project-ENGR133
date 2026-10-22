"""
Course Number: ENGR 13300
Semester: e.g. Spring 2026

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     8.2.1. Task 1 (for Python 1 Team task 1)
    Team ID:        LC9 - 09 (e.g. LC1 - 01; for section LC1, team 01)
    Author:         Adam Rose, rose333@purdue.edu
    Date:           e.g. 01/23/2026

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def rgb_to_grayscale(rgb_array):
    return np.dot(rgb_array[..., :3], [0.2126, 0.7152, 0.0722])

# Scale image values from 0-255 to 0-1 for processing.
def normalize_img(img_array):
    return img_array / 255.0

# Linearize sRGB values before numeric processing.
def linearize_img(norm_array):
    rdtype = np.asarray(norm_array).dtype
    values = np.asarray(norm_array, dtype=np.float64)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    return linear.astype(rdtype)

# Load the image and strip the alpha channel if present.
def load_img(filename):
    img = Image.open(filename)
    img_arr = np.array(img)

    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
            img_arr = img_arr[..., :3]

    return linearize_img(normalize_img(img_arr))

def apply_sobel_x(img_array):
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    return convolve2d(img_array, sobel_x)

def main():
    filePath = input("Enter the path to the image file: ")
    img_arr = load_img(filePath)

    gray_arr = rgb_to_grayscale(img_arr)


if __name__ == "__main__":
    main()
