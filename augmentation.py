import os
import cv2
import albumentations as A
from albumentations.core.composition import OneOf

data_dir = "Dataset/train"

# Define the augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=1.0),  # Flip image horizontally
    A.RandomBrightnessContrast(p=0.6),  # Adjust brightness and contrast
    A.Rotate(limit=30, p=0.7),  # Rotate image within [-30, 30] degrees
    OneOf([
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.MotionBlur(blur_limit=3, p=0.4)
    ], p=0.3),
])

# Loop through each class folder
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        # Loop through each image in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Load the image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping {img_path}, unable to read.")
                continue

            # Apply augmentations
            augmented = augment(image=image)
            augmented_image = augmented["image"]

            # Create a new filename for the augmented image
            new_img_name = f"aug_{img_name}"
            new_img_path = os.path.join(class_path, new_img_name)

            # Save the augmented image
            cv2.imwrite(new_img_path, augmented_image)
            print(f"Saved: {new_img_path}")
