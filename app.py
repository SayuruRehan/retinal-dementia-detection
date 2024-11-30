import sys
sys.path.append('unet-model')
from model import attentionunet

import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from albumentations import Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, ElasticTransform

# Load the pre-trained segmentation model
from tensorflow.keras.models import load_model

# Load your segmentation model
input_shape = (256, 256, 1)

def load_segmentation_model(model_path):
    from model import attentionunet
    model = attentionunet(input_shape=input_shape)
    model.load_weights(model_path)
    return model

# Define augmentation pipeline
def augment_image_realistic(image):
    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ElasticTransform(alpha=1, sigma=50, p=0.3)  # Removed invalid 'alpha_affine'
    ])
    augmented = augmentation_pipeline(image=image)
    augmented_image = augmented['image']
    return augmented_image


# Preprocessing Function
def preprocess_image(image_path, target_size=256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_padded = pad_to_square(img_blur, target_size)
    img_normalized = img_padded / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=(0, -1))
    return img_normalized

def pad_to_square(img, target_size):
    height, width = img.shape[:2]
    max_dim = max(height, width)
    square_img = np.zeros((max_dim, max_dim), dtype=img.dtype)
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = img
    return cv2.resize(square_img, (target_size, target_size))

# Vascular Metrics Calculations
def arc_length_chord_length_ratio(vessel_segment):
    coords = np.column_stack(np.where(vessel_segment))
    arc_length = np.sum([euclidean(coords[i], coords[i+1]) for i in range(len(coords) - 1)])
    chord_length = euclidean(coords[0], coords[-1])
    return arc_length / chord_length if chord_length > 0 else 0

# Modify calculate_fractal_dimension to avoid log(0)
def calculate_fractal_dimension(skeleton, box_sizes=[2, 4, 8, 16, 32, 64]):
    try:
        if np.sum(skeleton) < 10:  # Ensure the skeleton is not too sparse
            return None

        box_counts = []
        for box_size in box_sizes:
            resized_img = cv2.resize(
                skeleton.astype(np.uint8),
                (skeleton.shape[1] // box_size, skeleton.shape[0] // box_size),
                interpolation=cv2.INTER_NEAREST
            )
            box_counts.append(np.sum(resized_img > 0))

        # Remove zeros from box_counts to avoid issues with log
        non_zero_indices = np.array(box_counts) > 0
        log_box_sizes = np.log(np.array(box_sizes)[non_zero_indices])
        log_box_counts = np.log(np.array(box_counts)[non_zero_indices])

        if len(log_box_sizes) < 2:  # Check if there are enough points for regression
            return None

        slope, _, _, _, _ = linregress(log_box_sizes, log_box_counts)
        return -slope
    except Exception as e:
        print(f"Error in fractal dimension calculation: {e}")
        return None


def calculate_cra_or_crv(diameters, constant):
    diameters = sorted(diameters, reverse=True)[:6]
    while len(diameters) > 1:
        combined = constant * np.sqrt(diameters[0]**2 + diameters[-1]**2)
        diameters = diameters[1:-1]
        diameters.append(combined)
    return diameters[0] if diameters else None

# Image Processing
def process_image(image_path, model, is_augmented=False, aug_index=None):
    preprocessed_img = preprocess_image(image_path)
    segmented_output = model.predict(preprocessed_img)[0, :, :, 0]
    skeleton = skeletonize(segmented_output > 0.5)
    labeled_skeleton = label(skeleton, connectivity=2)

    # Tortuosity and Diameter Metrics
    tortuosity_list = []
    diameters = []
    for i in range(1, labeled_skeleton.max() + 1):
        vessel_segment = (labeled_skeleton == i)
        tortuosity = arc_length_chord_length_ratio(vessel_segment)
        tortuosity_list.append(tortuosity)
        coords = np.column_stack(np.where(vessel_segment))
        if len(coords) > 1:
            max_diameter = np.linalg.norm(coords[-1] - coords[0])
            diameters.append(max_diameter)

    # Calculate CRAE/CRVE and AVR
    pixel_size = 4
    diameters_in_micrometers = [d * pixel_size for d in diameters]
    CRAE = calculate_cra_or_crv(diameters_in_micrometers, constant=0.88)
    CRVE = calculate_cra_or_crv(diameters_in_micrometers, constant=0.95)
    AVR = CRAE / CRVE if CRAE and CRVE else None

    # Calculate Fractal Dimension
    fractal_dimension = calculate_fractal_dimension(skeleton)

    # Generate identifier for augmented images
    image_identifier = f"{os.path.basename(image_path)}_aug{aug_index}" if is_augmented else os.path.basename(image_path)

    return {
        'Image': image_identifier,
        'Mean Tortuosity': np.mean(tortuosity_list) if tortuosity_list else None,
        'CRAE': CRAE,
        'CRVE': CRVE,
        'AVR': AVR,
        'Fractal Dimension': fractal_dimension,
    }

# Main Processing Function with Augmentation
def process_dataset_with_augmentation(base_path, model, output_csv, augmentations_per_image=5):
    results = []

    for condition in ['AD', 'Healthy']:
        condition_path = os.path.join(base_path, condition)
        print(f"Processing condition: {condition}, Path: {condition_path}")
        for subject in os.listdir(condition_path):
            subject_path = os.path.join(condition_path, subject)
            print(f"Processing subject: {subject}, Path: {subject_path}")

            for image_file in ['right_eye.jpg', 'left_eye.jpg']:
                image_path = os.path.join(subject_path, image_file)
                if os.path.exists(image_path):
                    print(f"Processing image: {image_file}")

                    # Process Original Image
                    parameters = process_image(image_path, model)
                    parameters['Condition'] = condition
                    parameters['Subject'] = subject
                    results.append(parameters)

                    # Perform Augmentations and Process Each Augmented Image
                    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    for aug_idx in range(augmentations_per_image):
                        augmented_image = augment_image_realistic(original_image)

                        # Save augmented image temporarily
                        augmented_image_path = f"/{subject}_{image_file}_aug{aug_idx}.jpg"
                        cv2.imwrite(augmented_image_path, augmented_image)

                        # Process augmented image
                        parameters_aug = process_image(augmented_image_path, model, is_augmented=True, aug_index=aug_idx)
                        parameters_aug['Condition'] = condition
                        parameters_aug['Subject'] = subject
                        results.append(parameters_aug)
                else:
                    print(f"Image not found: {image_file} in {subject_path}")

    # Save Results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}.")
    else:
        print("No results to save. Please check the dataset and processing pipeline.")

# Run the Pipeline
base_path = 'Data'
output_csv = 'unet-model/output/vascular_parameters_calculated.csv'
model_path = 'unet-model/Trained models/retina_attentionUnet_150epochs.hdf5'

model = load_segmentation_model(model_path)
process_dataset_with_augmentation(base_path, model, output_csv)