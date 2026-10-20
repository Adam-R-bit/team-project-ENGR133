"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Write two functions, one to build a Gaussian kernel
    and one to apply it, then use them to blur a user-selected
    grayscale image.

Assignment Information:
    Assignment:     Tp1 Team 3
    Team ID:        LC9_09
    Author:         Guen Batsaihan, gbatsaih@purdue.edu
    Date:           10/20/2026

Contributors:
    Hrishikesh Villur, hvillur@purdue.edu
    Adam Rose, rose333@purdue.edu
    Harlan Sigler, hsigler@purdue.edu

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
import numpy as np
import math
from pathlib import Path
from PIL import Image, ImageOps


"""
Function creates normalized 2D Gaussian kernel as NumPy array.
"""
def create_gaussian_kernel(sigma):
    # Radius of 3*sigma captures ~99.7% of the Gaussian's weight
    radius = int(np.ceil(3 * sigma))
    # Odd kernel size so there is a single center pixel
    kernel_size = 2 * radius + 1
 
    # Coordinate grids of each cell's offset from the center
    ax = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(ax, ax)
 
    # Apply 2D Gaussian formula; cells near the center get larger weights
    kernel = (1.0 / (2.0 * math.pi * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))
 
    # Normalize so weights sum to 1.0 (keeps overall brightness unchanged)
    kernel = kernel / np.sum(kernel)
    return kernel


"""
Function applies 2D Gaussian filter to grayscale image with zero-padding
and sliding window operation with stride = 1.
"""
def gaussian_filter(image, sigma):
    # Build kernel and get its size (kernel is square)
    kernel = create_gaussian_kernel(sigma)
    kernel_size = kernel.shape[0]
 
    # Pad by the kernel radius so the output matches the input size
    pad_size = (kernel_size - 1) // 2
 
    # Zero-pad the image (float avoids uint8 overflow in the sums)
    image_float = image.astype(float)
    padded_image = np.pad(image_float, pad_width=pad_size, mode='constant', constant_values=0.0)
 
    # Image dimensions and float array to hold the filtered results
    h, w = image.shape
    output = np.zeros_like(image, dtype=float)
 
    # Slide kernel across image with stride = 1 and compute weighted sums
    for i in range(h):          # Each output row
        for j in range(w):      # Each output column
            # Window centered on original pixel (i, j)
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Weighted sum of the window gives the blurred pixel value
            output[i, j] = np.sum(window * kernel)
 
    # Clip to 0-255 and truncate to uint8 (no rounding)
    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    # Ask user for image filename
    filename = input("Enter the path to the image file: ").strip()
 
    # Load image from ref_images folder, falling back to a direct path
    image_path = Path("ref_images") / filename
    if not image_path.exists():
        image_path = Path(filename)
 
    # Convert to grayscale NumPy array
    img = Image.open(image_path).convert('RGB')
    gray_img = np.array(ImageOps.grayscale(img), dtype=np.uint8)
 
    # Apply gaussian_filter with sigma = 1.0
    sigma = 1.0
    smoothed_img = gaussian_filter(gray_img, sigma)
 
    # Build output name, e.g. "cat.png" -> "cat_smooth.png"
    base_name = image_path.stem
    output_filename = f"{base_name}_smooth.png"
 
    # Convert back to a PIL image, then save and display it
    result_img = Image.fromarray(smoothed_img)
    result_img.save(output_filename)
    result_img.show()


if __name__ == "__main__":
    main()