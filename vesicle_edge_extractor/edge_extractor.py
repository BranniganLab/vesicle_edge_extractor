#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 12:27:16 2025

@author: js2746
"""
from scipy import ndimage
import cv2
from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np


def extract_edge_from_frame(frame, debug_path=None):
    """
    Extract vesicle edge from frame of vesicle video.

    Parameters
    ----------
    frame : numpy ndarray
        The 2D array of intensity values from a vesicle video frame.
    debug_path : pathlib Path, optional
        If not None, output debug images to debug_path. The default is None.

    Returns
    -------
    r_vals : numpy ndarray
        1D array of distances from center_of_mass to vesicle edge. Evenly spaced
        in theta from 0 to 2pi.
    center_of_mass : tuple
        The Cartesian coordinates of the approximate vesicle center.

    """
    # step 1: find internal vesicle point
    center_of_mass = approximate_vesicle_com(frame, debug_path=debug_path)

    # step 2: naive refinement of edge region
    polar_sobel, scaling_factor = wrap_image_to_polar(filters.sobel(frame), center_of_mass)
    avg = np.mean(np.argmax(polar_sobel, axis=1))
    vertically_masked_polar_sobel = isolate_region_of_array(polar_sobel, avg, 0.25)
    max_of_masked_region = np.argmax(vertically_masked_polar_sobel, axis=1)

    # step 3: FFT-informed refinement of edge region
    approx_edge = zero_out_all_but_lowest_n_modes(max_of_masked_region, n=7)

    # wrap original image to polar
    original_frame_polar, _ = wrap_image_to_polar(frame, center_of_mass)

    # step 4: horizontal Sobel filter and apply FFT-informed mask
    horizontal_sobel = filters.sobel(original_frame_polar, axis=1)
    gauss_blur = ndimage.gaussian_filter(horizontal_sobel, sigma=2)
    fft_masked_horizontal_sobel = isolate_region_of_array(gauss_blur, approx_edge, 0.05, True)
    max_sobel = np.nanargmax(fft_masked_horizontal_sobel, axis=1)

    r_vals = np.array(max_sobel) / scaling_factor

    return r_vals, center_of_mass


def make_debug_image(frame, output_path):
    """
    Make a debug image that shows each step in the process.

    Parameters
    ----------
    frame : numpy ndarray
        The 2D array of intensity values from a vesicle video frame.
    output_path : pathlib Path
        Output debug images to output_path.

    Returns
    -------
    None.

    """
    _, axes = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')
    plt.axis('off')
    axes[0].imshow(frame, cmap='gray')

    center_of_mass = approximate_vesicle_com(frame)
    polar_image, scaling_factor = wrap_image_to_polar(filters.sobel(frame), center_of_mass)
    axes[1].imshow(polar_image, cmap='gray')
    axes[0].scatter(center_of_mass[1], center_of_mass[0], color='tab:red')

    avg = np.mean(np.argmax(polar_image, axis=1))
    masked_polar_image = isolate_region_of_array(polar_image, avg, 0.35)

    max_of_masked_region = np.argmax(masked_polar_image, axis=1)
    axes[1].plot(max_of_masked_region, np.arange(0, max_of_masked_region.shape[0]))

    ifft = zero_out_all_but_lowest_n_modes(max_of_masked_region, n=7)
    axes[2].imshow(polar_image, cmap='gray')
    axes[2].plot(ifft, np.arange(0, polar_image.shape[0]), color='tab:blue')

    og_polar_image, _ = wrap_image_to_polar(frame, center_of_mass)
    horizontal_sobel = filters.sobel(og_polar_image, axis=1)
    gauss_blur = ndimage.gaussian_filter(horizontal_sobel, sigma=2)
    polar_image_masked = isolate_region_of_array(gauss_blur, ifft, 0.05, True)
    axes[3].imshow(polar_image_masked, cmap='gray')

    max_sobel = np.nanargmax(polar_image_masked, axis=1)
    axes[3].plot(max_sobel, np.arange(0, horizontal_sobel.shape[0]), color='red')

    r_vals = np.array(max_sobel) / scaling_factor
    x_vals, y_vals = convert_to_cartesian((center_of_mass[1], center_of_mass[0]), r_vals)
    axes[0].plot(x_vals, y_vals, color='red', alpha=.5)

    axes[0].set_title("Raw image")
    axes[1].set_title("Sobel filter; polar")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("theta (arbitrary units)")
    axes[2].set_title("Inverse FFT of first 7 modes")
    axes[2].set_xlabel("r")
    axes[2].set_ylabel("theta (arbitrary units)")
    axes[3].set_title("1D sobel of each row")
    axes[3].set_ylabel('theta (arbitrary units)')
    axes[3].set_xlabel('r')
    plt.savefig(output_path)
    plt.clf()
    plt.close()


def convert_to_cartesian(center_point, r_vals):
    """
    Convert r values to X and Y values.

    Parameters
    ----------
    center_point : tuple
        The origin (in X and Y) of your coordinate system.
    r_vals : list
        The r values to convert

    Returns
    -------
    list
        The X and Y values in cartesian space.

    """
    if not isinstance(r_vals, (list, np.ndarray)):
        raise TypeError("r_vals must be a list or 1D numpy array.")
    if isinstance(r_vals, np.ndarray):
        if len(r_vals.shape) != 1:
            raise TypeError("r_vals cannot have more dimensions than 1.")
        num_r = r_vals.shape[0]
    else:
        num_r = len(r_vals)
    origin_x, origin_y = center_point
    theta = np.linspace(0, 2 * np.pi, num_r)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_vals = r_vals * cos_theta + origin_x
    y_vals = r_vals * sin_theta + origin_y
    return x_vals, y_vals


def convert_to_polar(x_vals, y_vals, origin):
    """
    Convert Cartesian coordinates to radius values in polar coordinates.

    Parameters
    ----------
    x_vals : numpy ndarray
        The X coordinates.
    y_vals : numpy ndarray
        The Y coordinates.
    origin : tuple
        The X, Y coordinates of where the origin should be in the polar system.

    Returns
    -------
    numpy ndarray
        1D array of r values. Assumes evenly distributed points in theta from
        0 to 2pi.

    """
    x_vals = x_vals - origin[0]
    y_vals = y_vals - origin[1]
    return np.sqrt(x_vals ** 2 + y_vals ** 2)


def approximate_vesicle_com(frame, sigma=10, debug_path=None):
    """
    Find the center of mass of a vesicle within an image frame.

    Workflow: sobel filter -> gaussian blur -> otsu threshold -> find centroid.

    Parameters
    ----------
    frame : 2D numpy ndarray
        The image frame containing your vesicle.
    sigma : float, OPTIONAL
        Sigma value for the amount of gaussian blurring. Default is 10.
    debug_path : pathlib Path, OPTIONAL
        If not None, make an image that shows each step of this process and save\
        it to this path.

    Returns
    -------
    tuple
        The coordinates for the approximate center of mass of your vesicle.

    """
    sobel_filter = filters.sobel(frame)
    blurred = ndimage.gaussian_filter(sobel_filter, sigma=sigma)
    greater_than_otsu_threshold = (blurred > filters.threshold_otsu(blurred)).astype(int)
    centroid = regionprops(greater_than_otsu_threshold, blurred)[0].centroid
    if debug_path is not None:
        _make_debug_image_centroid(frame, sobel_filter, blurred, greater_than_otsu_threshold, centroid, debug_path)
    return centroid


def _approximate_vesicle_com_contrast_only(frame, threshold=.1, debug_path=None):
    """
    Find the center of mass of a vesicle within an image frame using contrast\
    method only.

    Parameters
    ----------
    frame : 2D numpy ndarray
        The image frame containing your vesicle.
    threshold : float, OPTIONAL
        Keep only top _ percent of intensity values. Default is 10%, (0.1).
    debug_path : pathlib Path, OPTIONAL
        If not None, save image to this path.

    Returns
    -------
    tuple
        The coordinates for the approximate center of mass of your vesicle.

    """
    maxval_threshold = (frame > np.amax(frame) - np.amax(frame) * threshold).astype(int)
    centroid = regionprops(maxval_threshold, frame)[0].centroid
    _, ax = plt.subplots()
    ax.imshow(maxval_threshold, cmap='jet')
    ax.scatter(centroid[1], centroid[0])
    if debug_path:
        plt.savefig(debug_path)
    return centroid


def wrap_image_to_polar(image, origin_coords):
    """
    Convert an image from cartesian to polar about an origin point.

    Parameters
    ----------
    image : 2D numpy ndarray
        The image you wish to convert to polar coordinates.
    origin_coords : 2-tuple
        The X and Y coordinates that represent the new image origin.

    Returns
    -------
    2D numpy ndarray
        The image, wrapped into polar coordinates.
    float
        The scaling factor needed to convert back to Cartesian coordinates.

    """
    max_r = np.sqrt(((image.shape[0] / 2.0) ** 2.0) + ((image.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(image, (origin_coords[1], origin_coords[0]), max_r, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
    scaling_factor = polar_image.shape[1] / max_r
    return polar_image, scaling_factor


def zero_out_all_but_lowest_n_modes(arr, n):
    """
    Take the FFT of a 1d array, remove the lowest n modes, then IFFT.

    Parameters
    ----------
    arr : 1D numpy ndarray or list
        The input array.
    n : int
        The highest mode you wish to retain.

    Raises
    ------
    TypeError
        n must be an int.
    ValueError
        n must be positive.
    IndexError
        n can't exceed the number of positive modes.

    Returns
    -------
    ifft : 1D numpy ndarray
        The original input array, but with high frequencies excluded.

    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be a positive integer")
    if n >= arr.shape[0] // 2:
        raise IndexError(f"arr does not have enough modes ({arr.shape[0]}) to zero out all but the lowest {n}.")
    fft = np.fft.fft(arr)
    fft[n + 1:-1 * n] = 0
    ifft = np.fft.ifft(fft)
    return ifft.real


def _make_debug_image_centroid(original, sobel, blur, threshold, centroid, fpath):
    """Make a 4-panel figure showing the steps in the centroid finding process."""
    _, axes = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Raw image')
    axes[0].scatter(centroid[1], centroid[0], color='tab:blue')

    axes[1].imshow(sobel, cmap='gray')
    axes[1].set_title('Sobel filter')

    axes[2].imshow(blur, cmap='gray')
    axes[2].set_title('Gaussian blur')

    axes[3].imshow(threshold, cmap='jet')
    axes[3].set_title('Otsu threshold')
    axes[3].scatter(centroid[1], centroid[0], color='tab:blue')

    plt.axis('off')
    plt.savefig(fpath.joinpath("centroid_process_debug.pdf"))


def isolate_region_of_array(arr, mask_center, threshold, set_bg_to_nan=False):
    """
    Use a mask to preserve region of array within threshold of mask_center.

    If mask_center is a scalar, preserve within threshold of mask_center on all
    rows. If mask_center is a list or ndarray, preserve within threshold of
    mask_center[i] on each row i.

    Parameters
    ----------
    arr : 2D numpy ndarray
        The original image/array that you wish to mask.
    mask_center : int, float, list or numpy ndarray
        The center column index from which to mask. If a scalar is provided,
        use a static mask over all rows. If a list or ndarray is provided, use
        a moving mask over each row.
    threshold : int
        How many bins on either side of mask_center to preserve.
    set_bg_to_nan : bool, OPTIONAL
        If True, set all values outside of mask to np.nan. If False, set all
        values outside of mask to 0.

    Raises
    ------
    IndexError
        If arr is not a 2D array or if arr and mask_center are different sizes
        along 0th dimension.
    TypeError
        If arr isn't a numpy array or if mask_center isn't an scalar, array or
        list.

    Returns
    -------
    masked_copy : 2D numpy ndarray
        The original arr with all values set to 0 or np.nan except those
        within the masked region.

    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy ndarray.")
    if len(arr.shape) != 2:
        raise IndexError("arr must be a 2D array")

    bg = 0
    if set_bg_to_nan:
        bg = np.nan
    masked_copy = np.full_like(arr, bg)

    if np.isscalar(mask_center):
        # static mask: preserve within threshold of mask_center for all rows.
        lower_bound = int(mask_center - mask_center * threshold)
        upper_bound = int(mask_center + mask_center * threshold) + 1
        masked_copy[:, lower_bound:upper_bound] = arr[:, lower_bound:upper_bound]
    elif isinstance(mask_center, (np.ndarray, list)):
        # moving mask: preserve within threshold of mask_center[i] on row i.
        if isinstance(mask_center, list):
            mask_center = np.array(mask_center)
        if mask_center.shape[0] != arr.shape[0]:
            raise IndexError("arr and mask_center must be same size in 0th dimension")
        for index, center_value in enumerate(mask_center):
            lower_bound = int(center_value - center_value * threshold)
            upper_bound = int(center_value + center_value * threshold) + 1
            masked_copy[index, lower_bound:upper_bound] = arr[index, lower_bound:upper_bound]
    else:
        raise TypeError("mask_center must be a scalar, list, or numpy ndarray.")

    return masked_copy
