import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--update-ref-values", 
        action="store_true", 
        default=False,
        help="Update reference values instead of asserting"
    )
