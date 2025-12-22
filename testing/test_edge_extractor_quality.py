#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:27:45 2025.

@author: js2746
"""

import pytest
import glob
import numpy as np
from vesicle_edge_extractor.vesicle_video import VesicleVideo
from vesicle_edge_extractor.edge_extractor import extract_edge_from_frame


@pytest.fixture(scope='module')
def all_sample_videos():
    video_list = []
    for file in glob.glob('./sample_vesicle_videos/*.npy'):
        video = VesicleVideo(np.load(file))
        video.extract_edges(extract_edge_from_frame)
        video_list.append(video)
    return video_list


def test_whether_files_all_saved(all_sample_videos):
    assert len(all_sample_videos) == 8, "One or more videos failed"


def test_whether_edges_extracted(all_sample_videos):
    for video in all_sample_videos:
        assert not np.isnan(video.x_vals).any(), "Some x_vals are nan"
