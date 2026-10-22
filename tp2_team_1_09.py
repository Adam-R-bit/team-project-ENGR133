"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Detects edges in a traffic sign image. The program loads a color
    image, linearizes it, converts it to grayscale, blurs it with a
    Gaussian filter (sigma = 1), then applies a Sobel filter: x and y
    gradients, combined gradient magnitude clipped to 255, and a
    threshold of 50. The resulting edge image is displayed and saved.

Assignment Information:
    Assignment:     8.2.1. Task 1 (TP2 Team Task 1)
    Team ID:        LC9 - 09
    Author:         Adam Rose, rose333@purdue.edu
    Date:           10/22/2026

Contributors:
    Hrishikesh Villur, login@purdue.edu
    Harlan Sigler, login@purdue.edu
    Guen Batsaihan, login@purdue.edu

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

import math
import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

SOBEL_THRESHOLD = 50

def normalize_img(img_array):
    """Scale integer pixel values from [0, 255] to [0.0, 1.0].
    Float values already in [0.0, 1.0] are left unchanged."""
    if np.issubdtype(img_array.dtype, np.integer):
        return img_array.astype(float) / 255.0
    return img_array.astype(float)


def linearize_img(norm_array):
    """Apply the sRGB transfer function to every channel value."""
    return np.where(
        norm_array <= 0.04045,
        norm_array / 12.92,
        ((norm_array + 0.055) / 1.055) ** 2.4,
    )


def load_img(file_path):
    """Load an image, drop the alpha channel if present, normalize it
    to [0, 1], and linearize it."""
    img_arr = np.array(Image.open(file_path))

    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    return linearize_img(normalize_img(img_arr))


def rgb_to_grayscale(rgb_array):
    """Convert a linearized RGB image to a 2D grayscale image using
    ITU-R BT.709 weights."""
    if rgb_array.ndim == 2:
        return rgb_array
    return (0.2126 * rgb_array[:, :, 0]
            + 0.7152 * rgb_array[:, :, 1]
            + 0.0722 * rgb_array[:, :, 2])

def filter_2d(image, kernel):
    """Zero-pad the image, slide the kernel across it with stride 1,
    and return the weighted sum at each location. The output has the
    same shape as the input, and the input is not modified."""
    image = np.array(image, dtype=float)
    kernel = np.array(kernel, dtype=float)

    k_size = kernel.shape[0]
    pad = (k_size - 1) // 2
    padded = np.pad(image, pad, mode="constant", constant_values=0)

    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=float)
    for i in range(k_size):
        for j in range(k_size):
            output += kernel[i, j] * padded[i:i + rows, j:j + cols]
    return output

def create_gaussian_kernel(sigma):
    """Build a square, odd-sized Gaussian kernel that sums to 1."""
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    center = kernel_size // 2

    offsets = np.arange(kernel_size) - center
    x, y = np.meshgrid(offsets, offsets)
    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(x ** 2 + y ** 2) / (2 * sigma ** 2)
    )
    return kernel / np.sum(kernel)


def gaussian_filter(image, sigma):
    """Blur a 2D grayscale image with a Gaussian kernel."""
    kernel = create_gaussian_kernel(sigma)
    return filter_2d(image, kernel)

def get_sobel_kernels():
    """Return the Sobel x and y kernels as float arrays."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=float)
    return sobel_x, sobel_y


def to_255_scale(image):
    """load_img returns values in [0, 1]. Scale to [0, 255] so the
    255 clip and the threshold of 50 apply."""
    return np.array(image, dtype=float) * 255.0


def compute_gradient_magnitude(grad_x, grad_y):
    """Combine the gradients pixel by pixel and clip values above 255."""
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.minimum(magnitude, 255)


def apply_threshold(magnitude, threshold):
    """Pixels >= threshold become 255; all others become 0."""
    return np.where(magnitude >= threshold, 255, 0)


def sobel_filter(blurred_image):
    """Detect edges in a blurred grayscale image. Returns a uint8 array."""
    image = to_255_scale(blurred_image)
    sobel_x, sobel_y = get_sobel_kernels()

    grad_x = filter_2d(image, sobel_x)
    grad_y = filter_2d(image, sobel_y)

    magnitude = compute_gradient_magnitude(grad_x, grad_y)
    edges = apply_threshold(magnitude, SOBEL_THRESHOLD)
    return edges.astype(np.uint8)

def main():
    file_path = input("Enter the path to the image file: ")

    img = load_img(file_path)
    gray = rgb_to_grayscale(img)
    blurred = gaussian_filter(gray, 1)
    edges = sobel_filter(blurred)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"{base_name}_edge.png"
    Image.fromarray(edges).save(output_path)

    plt.imshow(edges, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()