#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:27:45 2025.

@author: js2746
"""

import pytest
from pathlib import Path
import glob
import numpy as np
from vesicle_edge_extractor.vesicle_video import VesicleVideo
from vesicle_edge_extractor.edge_extractor import extract_edge_from_frame


def load_all_sample_videos():
    video_list = []
    name_list = []
    for file in glob.glob('./testing/sample_vesicle_videos/*.npy'):
        video = VesicleVideo(np.load(file))
        video.extract_edges(extract_edge_from_frame)
        video_list.append(video)
        name_list.append(Path(file).stem)
    return video_list, name_list


pytestmark = pytest.mark.parametrize("video,name", load_all_sample_videos())


def test_whether_edges_extracted(video, name):
    assert not np.isnan(video.x_vals).any(), f"Some x_vals are nan in {name}"


def test_filtering_success(video, name):
    hist = np.bincount(video.status)
    assert hist[0] == 0, f"Something didn't get filtered properly in {name}"


def test_extraction_quality(video, name):
    hist = np.bincount(video.status)
    assert hist[1] / np.sum(hist) > .67, f"Extraction rate below 67% in {name}"


