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


@pytest.fixture(scope="session")
def sample_videos():
    test_dir = Path(__file__).parent
    video_list = {}
    for path in test_dir.iterdir():
        if path.suffix == '.npy':
            video = VesicleVideo(np.load(path))
            video.extract_edges(extract_edge_from_frame)
            video_list[path.name] = video
    return video_list


def pytest_generate_tests(metafunc):
    if "filename" not in metafunc.fixturenames:
        return
    test_dir = Path(metafunc.definition.fspath).parent
    filenames = sorted(p.name for p in test_dir.iterdir() if p.suffix=='.npy')
    metafunc.parametrize("filename", filenames, ids=filenames)


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


