import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_random_images(dataset_path):
    """
    Load two random .jp2 images from the given dataset path.

    Args:
    dataset_path (str): Path to the folder containing .jp2 images.

    Returns:
    image1, image2: Two randomly selected images.
    """
    # Get all .jp2 files in the dataset path
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jp2')]
    
    # Randomly select two image files
    image1_file, image2_file = random.sample(image_files, 2)
    
    # Load the images
    image1 = cv2.imread(os.path.join(dataset_path, image1_file), cv2.IMREAD_COLOR)
    image2 = cv2.imread(os.path.join(dataset_path, image2_file), cv2.IMREAD_COLOR)
    
    return image1, image2

def compute_keypoints(image_pair):
    """
    Compute keypoints and descriptors for a pair of images using SIFT.

    Args:
    image_pair (tuple): A tuple containing two images (left and right).

    Returns:
    keypoints1, descriptors1, keypoints2, descriptors2: Keypoints and descriptors for the two images.
    """
    sift = cv2.SIFT_create()
    left = image_pair[0]
    right = image_pair[1]

    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    keypoints1, descriptors1 = sift.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_right, None)

    return keypoints1, descriptors1, keypoints2, descriptors2

def match_images(descriptors1, descriptors2, trsh=0.8):
    """
    Match descriptors between two sets using the BFMatcher with knnMatch and ratio test.

    Args:
    descriptors1 (ndarray): Descriptors from the first image.
    descriptors2 (ndarray): Descriptors from the second image.
    trsh (float): The ratio threshold for good matches (default 0.8).

    Returns:
    good_matches (list): A list of good matches based on the ratio test.
    """
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < trsh * n.distance:
            good_matches.append(m)

    return good_matches

def ransac_filter(matches, keypoints1, keypoints2, threshold=5.0):
    """
    Filter matches using RANSAC to remove outliers by computing homography.

    Args:
    matches (list): List of good matches.
    keypoints1 (list): List of keypoints for the first image.
    keypoints2 (list): List of keypoints for the second image.
    threshold (float): Threshold for RANSAC algorithm (default 5.0).

    Returns:
    inliers (list): List of inlier matches.
    outliers (list): List of outlier matches.
    """
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return [], []

    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    _, mask = cv2.findHomography(points1, points2, cv2.RANSAC, threshold)

    inliers = [m for i, m in enumerate(matches) if mask[i]]
    outliers = [m for i, m in enumerate(matches) if not mask[i]]

    return inliers, outliers

def visualize_keypoints_matches(image1, image2, keypoints1, keypoints2, inliers, outliers):
    """
    Visualize the matches, distinguishing inliers and outliers using different colors.
    """
    inlier_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inliers, None,
                                   matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    outlier_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, outliers, None,
                                    matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    combined_image = cv2.addWeighted(inlier_image, 0.5, outlier_image, 0.5, 0)
    plt.imshow(combined_image)
    plt.show()

def image_matching_algorithm(dataset_path):
    """
    Perform image matching on two randomly selected images from the dataset.
    
    Args:
    dataset_path (str): Path to the folder containing images.
    """
    # Load random images from the dataset
    image1, image2 = load_random_images(dataset_path)

    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1, keypoints2, descriptors2 = compute_keypoints((image1, image2))

    # Match the descriptors
    good_matches = match_images(descriptors1, descriptors2)

    # Filter matches using RANSAC
    inliers, outliers = ransac_filter(good_matches, keypoints1, keypoints2)

    # Visualize keypoints and matches
    visualize_keypoints_matches(
        image1,
        image2,
        keypoints1,
        keypoints2,
        inliers,
        outliers
    )
