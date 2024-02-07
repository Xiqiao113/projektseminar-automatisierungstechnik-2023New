# Python file that uses gaussian blur from openCV to blur heightmaps to get smoother terrains

import cv2
import matplotlib.pyplot as plt
import sys
import os

filepath = 'C:/Users/organ/Animation Rover/Assets'
sys.path.append(filepath)

# Read the image
image_path = ['Map1.jpg', 'Map2.jpg', 'Map3.jpg', 'Map4.jpg', 'Map5.jpg', 'Map6.jpg']
output_image_path = ['blurrMap1.jpg', 'blurrMap2.jpg', 'blurrMap3.jpg', 'blurrMap4.jpg', 'blurrMap5.jpg',
                     'blurrMap6.jpg']

for i in range(6):
    image = cv2.imread(image_path[i])

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load the image from {image_path[i]}")
        exit()

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur
    ksize = (5, 5)  # Kernel size (odd numbers)
    sigma = 0  # Standard deviation, 0 means it will be computed based on the kernel size
    blurred_image = cv2.GaussianBlur(image_rgb, ksize, sigma)

    # Save the blurred image
    output_path = os.path.join(filepath, output_image_path[i])
    cv2.imwrite(output_path, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))

    # Plot the original and blurred images
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Blurred Image')
    plt.imshow(blurred_image)
    plt.axis('off')

    plt.show()

    print(f"Blurred image saved to: {output_path}")
