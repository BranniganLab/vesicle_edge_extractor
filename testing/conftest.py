import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-ref-values", 
        action="store_true", 
        default=False,
        help="Update reference values instead of asserting"
    )


def pytest_collection_modifyitems(config, items):
    """
    If --update-ref-values flag supplied to pytest, skip everything except the
    fixture and test you need to update the reference values.
    """
    update = config.getoption("--update-ref-values")
    if not update:
        return
    
    skip_marker = pytest.mark.skip(
        reason = "Skipped because --update-ref-vales is True"
    )

    for item in items:
        if item.nodeid not in [
            "testing/test_edge_extractor_quality.py::sample_videos",
            "testing/test_edge_extractor_quality.py::test_extraction_quality",
        ]:
            item.add_marker(skip_marker)
