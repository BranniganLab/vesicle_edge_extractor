#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:42:28 2025

@author: js2746
"""
from dataclasses import dataclass, field
from pathlib import Path
import glob
import nd2
from scipy import ndimage
import cv2
from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class VesicleVideo:
    """
    A class for vesicle videos.

    Holds both the raw images of a vesicle video, as well as its computed edges.

    Attributes
    ----------
        frames : numpy ndarray
            The 3D array of raw images. 0th dimension is frame number.
        vesicle_centers : list of tuples
            List of len(frames.shape[0]) containing Cartesian coordinates of the
            approximate vesicle center for each frame. Needed for wrapping images
            to/from polar coordinates.
        r_vals : numpy ndarray
            The distance from vesicle_center on frame i to the edge of the vesicle
            on frame i. Evenly spaced in theta, ranging from 0 to 2pi.
        x_vals, y_vals : numpy ndarrays
            The Cartesian coordinates of the vesicle edge.
    """

    frames: np.ndarray
    vesicle_centers: list = field(init=False)
    r_vals: np.ndarray = field(init=False)
    x_vals: np.ndarray = field(init=False)
    y_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Do argument validation on frames. Initialize all else to None or nan.

        Raises
        ------
        TypeError
            If frames is not an ndarray.
        IndexError
            If frames is not a 3D ndarray.

        Returns
        -------
        None.

        """
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray")
        if len(self.frames.shape) != 3:
            raise IndexError("frames must be a 3D array")
        self.vesicle_centers = [None] * self.frames.shape[0]
        self.r_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)
        self.x_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)
        self.y_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)

    def make_vesicle_gif(self, path, trace=True):
        """
        Make a .gif of the vesicle, with or without the detected edges shown.

        Parameters
        ----------
        path : pathlib Path
            The location and filename to save this .gif to.
        trace : Bool, optional
            Whether or not to display the detected edges. The default is True.

        Raises
        ------
        ValueError
            If trace is True, but there are no edges saved.

        Returns
        -------
        None.

        """
        if not isinstance(path, Path):
            path = Path(path).resolve()
        if (trace and not np.isnan(self.x_vals[0]).any()):
            raise ValueError("trace was set to True, but there are no edges detected for this vesicle.")
        output_path = path.with_suffix('.gif')
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"frame {i} / {self.frames.shape[0]}")
            ax.imshow(self.frames[i], cmap='gray', animated='True')
            if trace:
                ax.plot(self.x_vals[i], self.y_vals[i])

        ani = FuncAnimation(fig, animate, frames=self.frames.shape[0] - 1, interval=150, blit=False, repeat_delay=1000)
        ani.save(output_path)
        plt.close()

    def add_edge_from_frame(self, r_vals, frame_num, vesicle_center):
        """
        Save detected edge information for a given frame.

        Parameters
        ----------
        r_vals : list or numpy ndarray
            The list or 1D array of radial distances from the vesicle_center,
            spaced evenly from 0 to 2pi.
        frame_num : int
            The frame number.
        vesicle_center : tuple
            The origin (in x, y) of the polar coordinate system.

        Returns
        -------
        None.

        """
        self.r_vals[frame_num] = r_vals
        self.vesicle_centers[frame_num] = vesicle_center
        self.x_vals[frame_num], self.y_vals[frame_num] = convert_to_cartesian((vesicle_center[1], vesicle_center[0],), r_vals)


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
    fig, ax = plt.subplots()
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
    elif n >= arr.shape[0] // 2:
        raise IndexError(f"arr does not have enough modes ({arr.shape[0]}) to zero out all but the lowest {n}.")
    fft = np.fft.fft(arr)
    fft[n + 1:-1 * n] = 0
    ifft = np.fft.ifft(fft)
    return ifft.real


def _make_debug_image_centroid(original, sobel, blur, threshold, centroid, fpath):
    """Make a 4-panel figure showing the steps in the centroid finding process."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')

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


def extract_edge(frame, debug_path=None):
    # step 1: find internal vesicle point
    center_of_mass = approximate_vesicle_com(frame, debug_path=debug_path)

    # step 2: naive refinement of edge region
    sobel = filters.sobel(frame)
    polar_sobel, scaling_factor = wrap_image_to_polar(sobel, center_of_mass)
    max_list = np.argmax(polar_sobel, axis=1)
    avg = np.mean(max_list)
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


def make_test_image(intensities, frame_num):
    frame = intensities[frame_num, :, :]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')
    plt.axis('off')
    axes[0].imshow(frame, cmap='gray')

    sobel = filters.sobel(frame)
    center_of_mass = approximate_vesicle_com(frame)
    polar_image, scaling_factor = wrap_image_to_polar(sobel, center_of_mass)
    axes[1].imshow(polar_image, cmap='gray')
    axes[0].scatter(center_of_mass[1], center_of_mass[0], color='tab:red')

    max_list = np.argmax(polar_image, axis=1)

    avg = np.mean(max_list)
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
    com_rev = (center_of_mass[1], center_of_mass[0])
    x_vals, y_vals = convert_to_cartesian(com_rev, r_vals)
    axes[0].plot(x_vals, y_vals, color='red', alpha=.5)

    new_com_x, new_com_y = np.mean(x_vals), np.mean(y_vals)
    axes[0].scatter(new_com_x, new_com_y, color='tab:blue')

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
    plt.savefig("/home/js2746/Desktop/test_image5.pdf")
    plt.clf()
    plt.close()


for file in glob.glob('/home/js2746/DOPC_TF/DOPC_C*/C*/11.5.25/*.nd2'):
    path = Path(file)
    print(f"working on file {path.stem}")
    if path.with_suffix(".npy").exists():
        continue
    intensities = nd2.imread(path)
    ves_vid = VesicleVideo(intensities)

    for frame_num, _ in enumerate(intensities):
        try:
            r_vals, vesicle_center = extract_edge(intensities[frame_num, :, :])
        except ValueError:
            r_vals = np.full_like(r_vals, np.nan)
            vesicle_center = (np.nan, np.nan)
        ves_vid.add_edge_from_frame(r_vals, frame_num, vesicle_center)

    ves_vid.make_vesicle_gif(path, True)
    np.save(path.with_suffix(".npy"), ves_vid.r_vals)
