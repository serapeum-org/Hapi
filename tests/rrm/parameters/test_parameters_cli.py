import subprocess
from unittest.mock import MagicMock, patch

import pytest

from hapi.parameters.parameters import main


@pytest.fixture
def mock_parameter_class():
    """Mock the Parameter class methods."""
    with patch("hapi.parameters.parameters.Parameter") as MockParameter:
        MockParameter.return_value = MagicMock()
        MockParameter.get_parameters.return_value = MagicMock()
        MockParameter.get_parameter_set.return_value = MagicMock()
        # MockParameter.list_parameter_names.return_value = MagicMock()
        yield MockParameter


def test_download_parameters(mock_parameter_class):
    """Test the download-parameters command."""

    with patch(
        "sys.argv",
        [
            "parameters.py",
            "download-parameters",
            "--directory",
            "/mock/dir",
            "--version",
            "2",
        ],
    ):
        main()

    mock_parameter_class.assert_called_once_with(version=2)
    mock_parameter_class.return_value.get_parameters.assert_called_once_with(
        download_dir="/mock/dir"
    )


def test_download_parameter_set(mock_parameter_class):
    """Test the download-parameter-set command."""
    with patch(
        "sys.argv",
        [
            "parameters.py",
            "download-parameter-set",
            "1",
            "--directory",
            "/mock/dir",
            "--version",
            "1",
        ],
    ):
        main()

    mock_parameter_class.assert_called_once_with(version=1)
    mock_parameter_class.return_value.get_parameter_set.assert_called_once_with(
        set_id="1", download_dir="/mock/dir"
    )


def test_list_parameter_names(mock_parameter_class):
    """Test the list-parameter-names command."""

    with patch("sys.argv", ["parameters.py", "list-parameter-names"]), patch(
        "hapi.parameters.parameters.Parameter.list_parameter_names",
        return_value=["01_tt", "02_rfcf", "03_sfcf"],
    ) as mock_list_names:
        main()

    # `list-parameter-names` doesn't instantiate `Parameter`, so ensure it's not called
    mock_parameter_class.assert_not_called()
    mock_list_names.assert_called_once()
