#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:27:45 2025.

@author: js2746
"""

import pytest
import math
from pathlib import Path
import json
import numpy as np
from vesicle_edge_extractor.vesicle_video import VesicleVideo
from vesicle_edge_extractor.edge_extractor import extract_edge_from_frame


# ----------------------------------------------------------
# Fixture: Expensive processing of all videos only done once
# ----------------------------------------------------------
@pytest.fixture(scope="session")
def sample_videos():
    """
    Load and process all files in test_data directory once per test session.
    Returns a dict mapping filename -> processed content.
    """
    test_file_dir = Path(__file__).parent / "sample_vesicle_videos"

    # Defensive: fail loudly if directory does not exist
    if not test_file_dir.exists():
        pytest.fail(f"Test data directory does not exist: {test_file_dir}")

    video_list = {}
    for path in test_file_dir.iterdir():
        if path.suffix == '.npy':
            video = VesicleVideo(np.load(path))
            video.extract_edges(extract_edge_from_frame)
            video_list[path.name] = video

    if not video_list:
        pytest.fail(f"No files found in test directory: {test_file_dir}")

    return video_list


# -------------------------------------------------------------------
# Hook: Make each test run once for each file by using filename as a
# parametrized variable. DO NOT RENAME THIS FUNCTION OR ITS ARGUMENT!
# -------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    """
    Dynamically parameterize the test function with filenames.
    Only affects tests that request 'filename'.
    """
    if "filename" not in metafunc.fixturenames:
        return

    test_file_dir = Path(metafunc.definition.fspath).parent / "sample_vesicle_videos"

    filenames = sorted(p.name for p in test_file_dir.iterdir() if p.suffix=='.npy')

    if not filenames:
        pytest.fail(f"No files found to parameterize test: {test_file_dir}")

    metafunc.parametrize("filename", filenames, ids=filenames)


# ----------------------------
# Actual tests
# ----------------------------
def test_whether_edges_extracted(filename, sample_videos):
    """
    If edge extraction failed, some frame(s) would contain NaN values.
    Check for that.
    """
    video = sample_videos[filename]
    assert not np.isnan(video.x_vals).any(), f"Some x_vals are nan in {filename}"


def test_filtering_success(filename, sample_videos):
    """
    If something went wrong in the filtering step, some frame(s) would have a
    status of 0 (as opposed to 1, 2, or 3). Check for that.
    """
    video = sample_videos[filename]
    hist = np.bincount(video.status)
    assert hist[0] == 0, f"Something didn't get filtered properly in {filename}"


def test_extraction_quality(filename, sample_videos):
    video = sample_videos[filename]
    hist = np.bincount(video.status)
    meas_pct_usbl_frames = hist[1] / np.sum(hist)
    
    expected_value_file = Path(__file__).parent / "reference_values" /  f"expected_value_{filename}.json"
    if not expected_value_file.is_file():
        pytest.skip(f"No reference data to compare against for file {filename}")

    with open(expected_value_file) as f:
        saved_data = json.load(f)

    exp_pct_usbl_frames = saved_data["expected pct useable value"]

    assert math.is_close(
        meas_pct_usbl_frames, exp_pct_usbl_frames, 0.01
    ), (
        f"Extraction rate does not match reference value for {filename}"
    )


