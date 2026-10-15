"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Loads an RGB image, standardizes it to 100x100, prints the RGB values
    at a user-specified pixel, converts the image to HSV, prints the HSV
    values at the same pixel, and displays the HSV image.

Assignment Information:
    Assignment:     7.2.1 TP1 Team 1
    Team ID:        LC9 - 09
    Author:         Adam Rose, rose333@purdue.edu
    Date:           10/15/2026

Contributors:
    Hrishikesh Villur, hvillur@purdue.edu
    Harlan Sigler, hsigler@purdue.edu
    Guen Batsaihan, gbatsaih@purdue.edu

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
from PIL import ImageOps, Image
import matplotlib.pyplot as plt


# Convert one RGB pixel (0-255 ints) to HSV, each scaled to 0-255 as floats.
def rgb_to_hsv(R, G, B):
    R = R / 255
    G = G / 255
    B = B / 255

    cMax = max(R, G, B)
    cMin = min(R, G, B)
    delta = cMax - cMin

    # Hue depends on which channel is largest; grayscale gets 0
    if delta == 0:
        H = 0
    elif cMax == R:
        H = (60 * (G - B) / delta) % 360
    elif cMax == G:
        H = (60 * (B - R) / delta + 120) % 360
    elif cMax == B:
        H = (60 * (R - G) / delta + 240) % 360

    S = 0 if cMax == 0 else delta / cMax

    V = cMax

    # Rescale H from degrees and S, V from [0, 1] to [0, 255]
    H = H / 360 * 255
    S = S * 255
    V = V * 255

    return H, S, V

# Apply rgb_to_hsv to every pixel; assigning into the uint8 array truncates values.
def convert_to_hsv(rgb_image):
    for row in range(rgb_image.shape[0]):
        for col in range(rgb_image.shape[1]):
            rgb_image[row][col] = rgb_to_hsv(rgb_image[row][col][0], rgb_image[row][col][1], rgb_image[row][col][2])

    return rgb_image.astype(np.uint8)

# Resize to new_size (width, height) with bilinear interpolation.
def resize_image(original_image_array, new_size):
    pil_img = Image.fromarray(original_image_array)
    resized_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)
    return np.array(resized_img)

# Pad to target_size (width, height) with black borders.
def pad_image(resized_img_array, target_size):
    pil_img = Image.fromarray(resized_img_array)
    padded_img = ImageOps.pad(pil_img,target_size,color=0)
    return np.array(padded_img)

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

# Load an image, drop the alpha channel if present, then normalize and linearize.
def load_img(filename):
    img = Image.open(filename)
    img_arr = np.array(img)

    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
            img_arr = img_arr[..., :3]

    return linearize_img(normalize_img(img_arr))


def main():
    filePath = input("Enter the path of the RGB image you want to convert to hsv: ")
    # Rescale the 0-1 floats from load_img to uint8 for Pillow
    imageArray = (load_img(filePath) * 255).astype(np.uint8)

    # Standardize to 100x100: resize keeping aspect ratio, then pad
    target_w, target_h = 100, 100
    orig_h, orig_w = imageArray.shape[:2]
    orig_ar = orig_w / orig_h
    target_ar = target_w / target_h

    if orig_ar >= target_ar:
        new_size = (int(target_w / orig_ar), target_h)
    else:
        new_size = (target_w, int(target_h / orig_ar))

    imageArray = resize_image(imageArray, new_size)
    imageArray = pad_image(imageArray, (target_w, target_h))

    coords = [int(x) for x in input("Enter the coordinates of the pixel you want inspect: ").split(", ")]
    print(f"RGB values of the ({coords[0]}, {coords[1]}) pixel: R={imageArray[coords[0]][coords[1]][0]}, G={imageArray[coords[0]][coords[1]][1]}, B={imageArray[coords[0]][coords[1]][2]}")

    print(f"Converting {filePath} to HSV...")
    imageArray = convert_to_hsv(imageArray)

    print(f"HSV values of the ({coords[0]}, {coords[1]}) pixel: H={imageArray[coords[0]][coords[1]][0]}, S={imageArray[coords[0]][coords[1]][1]}, V={imageArray[coords[0]][coords[1]][2]}")

    plt.imshow(imageArray)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()