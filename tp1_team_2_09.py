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
from pathlib import Path
from PIL import Image, ImageOps

def create_mean_kernel(kernel_size: int):
    value = 1.0 / (kernel_size * kernel_size)
    return np.full((kernel_size,kernel_size),value,dtype = float)

def apply_kernel(image, kernel):
    #zero-pad
    padded_img = pad_image(image, 1)
    #iterate through function
    for y in range(image.shape[0] / kernel.shape[0]):
        for x in range(image.shape[1] / kernel.shape[1]):
            window = image.reshape(kernel.shape[0], kernel.shape[1])
            padded_img[x][y] = sum(np.dot(window, kernel)) / window.shape()[0]**2
    return padded_img

def pad_image(resized_img_array,target_size):
    pil_img = Image.fromarray(resized_img_array)
    padded_img = ImageOps.pad(pil_img,target_size,color=0)
    return np.array(padded_img)
    
def main():
    # User input for kernel size
    kernel_size = int(input("Enter kernel size: "))
    kernel = create_mean_kernel(kernel_size) # Creates kernel array
    kernel_sum = np.sum(kernel)
    filtered_image = apply_kernel(image,kernel)
    
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print(f"Kernel sum: {kernel_sum:.6f}")
    print("Kernel:")
    print(kernel)
    print("Filtered image:")
    print(filtered_image)


if __name__ == "__main__":
    main()