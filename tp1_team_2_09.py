"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Write two functions, one to create a mean kernel and one to
    apply it

Assignment Information:
    Assignment:     7.2.2. Task 2
    Team ID:        LC9
    Author:         Hrishikesh Villur, hvillur@purdue.edu
    Date:           10/15/2026

Contributors:
    Adam Rose, rose333@purdue.edu
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
from PIL import Image


def create_mean_kernel(kernel_size):
    value = 1.0 / (kernel_size * kernel_size)
    return np.full((kernel_size, kernel_size), value, dtype=float)


def pad_image(resized_img_array, target_size):
    pil_img = Image.fromarray(resized_img_array.astype(np.uint8))
    padded_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(padded_img, dtype=float)


def apply_kernel(image, kernel):
    height = kernel.shape[0]
    width = kernel.shape[1]

    # pad to (image size + kernel size - 1) so every window fits
    padded_img = pad_image(image, image.shape[0] + height - 1)
    filtered_img = np.zeros(image.shape, dtype=float)

    # slide kernel across image with stride 1
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = padded_img[y:y + height, x:x + width]
            filtered_img[y, x] = np.sum(window * kernel)

    return filtered_img


def main():
    kernel_size = int(input("Enter kernel size: "))
    kernel = create_mean_kernel(kernel_size)
    kernel_sum = np.sum(kernel)

    image = np.zeros((5, 5))
    image[2, 2] = 255

    filtered_image = apply_kernel(image, kernel)

    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print(f"Kernel sum: {kernel_sum:.6f}")
    print("Kernel:")
    print(np.array2string(kernel, precision=6))
    print("Filtered image:")
    print(np.array2string(filtered_image, precision=6))


if __name__ == "__main__":
    main()