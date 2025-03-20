import os
from PIL import Image

# Define input and output directories
input_dir = 'ICASSP2025_Dataset/Inputs/Task_2_ICASSP'
output_dir = 'ICASSP2025_Dataset/Outputs/Task_2_ICASSP'
augmented_input_dir = 'ICASSP2025_Dataset/Inputs/Task_2_ICASSP_Augmented_Inputs'
augmented_output_dir = 'ICASSP2025_Dataset/Outputs/Task_2_ICASSP_Augmented_Outputs'

# Create output directories if they don't exist
os.makedirs(augmented_input_dir, exist_ok=True)
os.makedirs(augmented_output_dir, exist_ok=True)

# Function to apply augmentations
def augment_image(image, filename, output_folder):
    # Save original image
    image.save(os.path.join(output_folder, f'{filename}_original.png'))

    # 90-degree rotation
    rotated_90 = image.rotate(90, expand=True)  # Using expand=True to keep original size
    rotated_90.save(os.path.join(output_folder, f'{filename}_rotated_90.png'))

    # 180-degree rotation
    rotated_180 = image.rotate(180, expand=True)  # Using expand=True to keep original size
    rotated_180.save(os.path.join(output_folder, f'{filename}_rotated_180.png'))

    # 270-degree rotation
    rotated_270 = image.rotate(270, expand=True)  # Using expand=True to keep original size
    rotated_270.save(os.path.join(output_folder, f'{filename}_rotated_270.png'))

    # Horizontal flip
    flipped_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_horizontal.save(os.path.join(output_folder, f'{filename}_flipped_horizontal.png'))

    # Vertical flip
    flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_vertical.save(os.path.join(output_folder, f'{filename}_flipped_vertical.png'))

    # Horizontal flip + 90-degree rotation
    flipped_horizontal_rotated_90 = flipped_horizontal.rotate(90, expand=True)  # Using expand=True
    flipped_horizontal_rotated_90.save(os.path.join(output_folder, f'{filename}_flipped_horizontal_rotated_90.png'))

    # Vertical flip + 90-degree rotation
    flipped_vertical_rotated_90 = flipped_vertical.rotate(90, expand=True)  # Using expand=True
    flipped_vertical_rotated_90.save(os.path.join(output_folder, f'{filename}_flipped_vertical_rotated_90.png'))

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions if necessary
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        # Augment input images
        with Image.open(input_image_path) as input_img:
            augment_image(input_img, os.path.splitext(filename)[0], augmented_input_dir)

        # Augment corresponding output images
        with Image.open(output_image_path) as output_img:
            augment_image(output_img, os.path.splitext(filename)[0], augmented_output_dir)

print(f"Augmented images saved in {augmented_input_dir} and {augmented_output_dir}")
