################################################################################
#                                                                              #
# ARDD optic disc and cup segmentations segmentation utils.                    #
# Created by Thomas Chia and Cindy Wu                                          #
# Medical Research by Sreya Devarakonda                                        #
# Created for the 2021 Congressional App Challenge                             #
# Winning "webapp" of Virginia's 11th District                                 #
#                                                                              #
# Citations:                                                                   #
#   Fu Et. al  "Joint Optic Disc and Cup Segmentation Based                    #
#               on Multi-label Deep Network and Polar Transformation"          #
#               https://arxiv.org/abs/1801.00926                               #
# Special Notes:                                                               #
#   New threshold function, new image resizing, new CDR calculation            #
#                                                                              #
################################################################################

import os
import numpy as np
import tensorflow as tf 
import cv2 as cv2

from PIL import Image
from glaucoma_segmentation.post_process import imresize
from tensorflow.keras.preprocessing import image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize

DC_ROI_DIMS = 600 
DC_SEG_DIMS = 640
DC_CDR_DIMS = 400

def segment_image(crop_model, seg_model, DC_DATA_IMAGE): 
    """
    Main function that segments the image, does so in two main stages.

    Parameters:
        crop_model (str): Path to the Tensorflow model for optic disc crop.
        seg_model (str): Path to the Tensorflow model for optic disc segmentation.
    
    Returns:
        An image that has been semented.
    """
    # Image preprocessing
    original_image = np.asarray(image.load_img(DC_DATA_IMAGE))
    testing_image = resize(original_image, (DC_SEG_DIMS, DC_SEG_DIMS, 3)) * 255
    testing_image = np.reshape(testing_image, (1,) + testing_image.shape)
    
    # Run the model on preprocessed image   
    disc_map = crop_model.predict([testing_image])

    # Threshold the disk map by 0.5 threshold to make it a binary mask
    disc_map = threshold_image(np.reshape(disc_map, (DC_SEG_DIMS, DC_SEG_DIMS)), 0.5)

    # Get the center of the optic disk for image segmentation
    disc_region = regionprops(label(disc_map))
    disk_center_x = int(disc_region[0].centroid[0] * original_image.shape[0] / DC_SEG_DIMS)
    disk_center_y = int(disc_region[0].centroid[1] * original_image.shape[1] / DC_SEG_DIMS)
    
    # Crop out the optic disc for the image
    disc_region, err_xy, crop_xy = disc_crop(original_image, DC_ROI_DIMS, disk_center_x, disk_center_y)


    rectangular_form_image = rotate(cv2.linearPolar(disc_region, (DC_ROI_DIMS / 2, DC_ROI_DIMS / 2),
                                       DC_ROI_DIMS / 2, cv2.WARP_FILL_OUTLIERS), -90)

    test_image = image_resizing(rectangular_form_image, DC_CDR_DIMS)
    test_image = np.reshape(test_image, (1,) + test_image.shape)
    [_, _, _, _, prob_10] = seg_model.predict(test_image)

    # Run post-processing to extract mask
    pixel_probabilities = np.reshape(prob_10, (prob_10.shape[1], prob_10.shape[2], prob_10.shape[3]))
    disc_mask = np.array(Image.fromarray(pixel_probabilities[:, :, 0]).resize((DC_ROI_DIMS, DC_ROI_DIMS)))
    ocup_mask = np.array(Image.fromarray(pixel_probabilities[:, :, 1]).resize((DC_ROI_DIMS, DC_ROI_DIMS)))
    
    # Threshold the masks
    disc_mask[-round(DC_ROI_DIMS / 3):, :] = 0
    ocup_mask[-round(DC_ROI_DIMS / 2.):, :] = 0
    
    # Change the mask from rectangular to polar
    disc_mask = cv2.linearPolar(rotate(disc_mask, 90), (DC_ROI_DIMS / 2, DC_ROI_DIMS / 2),
                                DC_ROI_DIMS / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    ocup_mask = cv2.linearPolar(rotate(ocup_mask, 90), (DC_ROI_DIMS / 2, DC_ROI_DIMS / 2),
                                DC_ROI_DIMS / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    disc_mask = np.array(threshold_image(disc_mask, 0.5), dtype=int)
    ocup_mask = np.array(threshold_image(ocup_mask, 0.5), dtype=int)

    
    # RGB mask
    color_mask = np.zeros((DC_ROI_DIMS, DC_ROI_DIMS, 3)) # Creates a blank array to hold the RGB Mask
    color_mask[..., 0] = threshold_image(disc_mask, 0.5)
    color_mask[..., 1] = threshold_image(ocup_mask, 0.5)

    # Save the raw mask
    roi_final = np.array(threshold_image(ocup_mask, 0.5), dtype=int) + np.array(threshold_image(disc_mask, 0.5), dtype=int)
    Img_result = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.int8)
    Img_result[crop_xy[0]:crop_xy[1], crop_xy[2]:crop_xy[3], ] = roi_final[err_xy[0]:err_xy[1], err_xy[2]:err_xy[3], ]
    save_result = Image.fromarray((Img_result * 127).astype(np.uint8))

    cdr = get_cdr(save_result)

    return save_result, cdr 

def get_cdr(input_image_array):
    """
    Function that calculates the vertical cup to disc ratio. 

    Parameters:
        input_image_array (arr): An array or list that will be calculated.

    Returns:
        The p-value for Glaucoma by calculating the cup to disc ratio.
    """

    # Change input into numpy array
    segmented_im = np.asarray(input_image_array)
    # Define the optic disk and optic cup masks
    bw_image = segmented_im[:, :]
    
    # Define variables that house the largest y value for OD and OC
    OD_max_y = 0
    OC_max_y = 0

    # Find the image dimensions to look for the largest column (y value)
    im_max_y = bw_image.shape[1]

    # Define variables that house the largest y value for OD and OC
    OD_min_y = im_max_y
    OC_min_y = im_max_y

    # Defines the current initial row (y value)
    y = 0
    # Loop through each row in the image
    for row in bw_image:
        # Loop through each point in the row
        for point in row:
            # Defines the current column (x value)
            x = 0
            # Sees if the point is an optic disk point
            if point > 100 & point < 150: 
                # If it is optic disk point catalogue its y value to find its maximum value
                if y >= OD_max_y:
                    OD_max_y = y
                # If it is optic disk point catalogue its y value to fine its minimum value
                if y <= OD_min_y:
                    OD_min_y = y
            # Sees if the point is an optic cup point
            if point > 200:
                # If it is optic cup point catalogue its y value to find its maximum value
                if y >= OC_max_y:
                    OC_max_y = y
                # If it is optic cup point catalogue its y value to fine its minimum value
                if y <= OC_min_y:
                    OC_min_y = y
            # Update the current column (x value)
            x = x + 1
        # Update the current row (y value)
        y = y + 1
        # Reset the current column to 0 for the next row
        x = 0
    
    # Subtract the max from the min to get diameter
    OD_diameter = OD_max_y - OD_min_y
    OC_diameter = OC_max_y - OC_min_y
    # Divide to the get the CDR 
    CDR = OC_diameter/OD_diameter

    CDR = round(CDR, 3)
    
    return CDR
    

def image_resizing(input_image, input_size):
    """
    Resizes the image.

    Parameters:
        input_image (arr): The input image array.
        input_size (float): The size of the input image array.
    
    Returns:
        The resized image.
    """
    # Change image from list to array
    image = np.asarray(input_image).astype('float32')
    # Convert to rgb and resize image
    image = imresize(image, (input_size, input_size, 3))
    return image

def threshold_image(input_image, threshold):
    """
    Thresholds an image.

    Parameters:
        input_image (array): The image array that will be thresholded.
        threshold (int): The threshold value to be used.
    
    Returns:
        A thresholded image.
    """

    if input_image.max() > threshold:
        binary = input_image > threshold
    else:
        binary = input_image > input_image.max() / 2.0

    labeled_image = label(binary)
    regions = regionprops(labeled_image)
    area_list = [region.area for region in regions]

    if area_list:
        idx_max = np.argmax(area_list)
        binary[labeled_image != idx_max + 1] = 0
    return binary_fill_holes(np.asarray(binary).astype(int))

def disc_crop(original_image, DC_ROI_DIMS, disk_center_x, disk_center_y):
    """
    Crops the optic region given the coordinates and dimensions.

    Parameters:
        original_image: An array or list, of optic disk
        DC_ROI_DIMS: Size of the output image
        disk_center_x: Center of the optic disk x
        disk_center_y: Center of the optic disk y
    """

    resized = int(DC_ROI_DIMS / 2)
    disc_region = np.zeros((DC_ROI_DIMS, DC_ROI_DIMS, 3), dtype=original_image.dtype)
    crop_coord = np.array([disk_center_x - resized, disk_center_x + resized, disk_center_y - resized, disk_center_y + resized], dtype=int)
    err_coord = [0, DC_ROI_DIMS, 0, DC_ROI_DIMS]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > original_image.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - original_image.shape[0])
        crop_coord[1] = original_image.shape[0]

    if crop_coord[3] > original_image.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - original_image.shape[1])
        crop_coord[3] = original_image.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = original_image[crop_coord[0]:crop_coord[1],crop_coord[2]:crop_coord[3],]
    return disc_region, err_coord, crop_coord

