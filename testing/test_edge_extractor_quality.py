#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:27:45 2025.

@author: js2746
"""

import pytest
from pathlib import Path
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
    video = sample_videos[filename]
    assert not np.isnan(video.x_vals).any(), f"Some x_vals are nan in {filename}"


def test_filtering_success(filename, sample_videos):
    video = sample_videos[filename]
    hist = np.bincount(video.status)
    assert hist[0] == 0, f"Something didn't get filtered properly in {filename}"


def test_extraction_quality(filename, sample_videos):
    video = sample_videos[filename]
    hist = np.bincount(video.status)
    assert hist[1] / np.sum(hist) > .67, f"Extraction rate below 67% in {filename}"


