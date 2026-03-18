import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from Hapi.parameters.parameters import (
    FigshareAPIClient,
    FileManager,
    Parameter,
    ParameterManager,
)


class TestFigshareAPIClient:

    def test_figshare_api_client_get_article(self):
        """
        Test an actual API call to Figshare's API to retrieve an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        response = client.send_request("GET", "articles/19999901")

        # Check basic response structure
        assert isinstance(response, dict), "Response should be a dictionary."
        assert "id" in response, "Response should include an 'id' key."
        assert (
            response["id"] == 19999901
        ), "The article ID should match the requested ID."

    def test_figshare_api_client_get_article_version(self):
        """
        Test an actual API call to retrieve a specific version of an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        response = client.get_article_version(article_id=19999901, version=1)

        # Check basic response structure
        assert isinstance(response, dict), "Response should be a dictionary."
        assert "id" in response, "Response should include an 'id' key."
        assert "version" in response, "Response should include a 'version' key."
        assert (
            response["version"] == 1
        ), "The version should match the requested version."

    def test_figshare_api_client_list_article_versions(self):
        """
        Test an actual API call to retrieve all available versions of an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        versions = client.list_article_versions(19999901)

        # Check basic response structure
        assert isinstance(versions, list), "Response should be a list."
        assert len(versions) > 0, "There should be at least one version."
        for version in versions:
            assert (
                "version" in version
            ), "Each version entry should include a 'version' key."

    def test_figshare_api_client_invalid_article(self):
        """
        Test how the API client handles an invalid article ID.
        """
        client = FigshareAPIClient()

        with pytest.raises(Exception):
            client.send_request("GET", "articles/0")

    def test_figshare_api_client_no_content(self):
        """
        Test the API client for an endpoint with no content.
        """
        client = FigshareAPIClient()
        response = client.send_request(
            "GET", "articles"
        )  # Assuming this endpoint exists but has no content

        assert response is not None, "Response should not be None."
        assert isinstance(response, list), "Response should be a list."


class TestFileManager:

    @pytest.fixture
    def temp_directory(self, tmp_path):
        """Fixture to create a temporary directory for testing."""
        return tmp_path

    def test_download_file(self, temp_directory):
        """Test that a directory is created if it doesn't exist."""
        new_dir = temp_directory / "new_subdir"
        file_path = new_dir / "example.txt"
        url = "https://www.example.com"

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake content"]
        mock_response.raise_for_status = MagicMock()

        with patch("Hapi.parameters.parameters.requests.get", return_value=mock_response):
            FileManager.download_file(url, file_path)

        assert new_dir.exists(), "The directory should be created."
        assert file_path.exists(), "The file should be created in the new directory."

    def test_clear_directory(self, temp_directory):
        """Test clearing all files in a directory."""
        # Create sample files
        for i in range(3):
            (temp_directory / f"file_{i}.txt").write_text(f"Content {i}")

        FileManager.clear_directory(temp_directory)

        assert not any(
            temp_directory.iterdir()
        ), "The directory should be empty after clearing."

    def test_clear_empty_directory(self, temp_directory):
        """Test clearing an already empty directory."""
        FileManager.clear_directory(temp_directory)

        assert not any(temp_directory.iterdir()), "The directory should remain empty."

    def test_clear_nonexistent_directory(self):
        """Test clearing a directory that doesn't exist."""
        non_existent_dir = Path("./non_existent_directory")

        FileManager.clear_directory(non_existent_dir)

        assert (
            not non_existent_dir.exists()
        ), "The directory should not exist and should not cause errors."


@pytest.mark.mock
class TestParameterManagerMock:

    @pytest.fixture
    def mock_api_client(self):
        """Fixture to provide a mock API client."""
        return MagicMock(spec=FigshareAPIClient)

    @pytest.fixture
    def parameter_manager(self, mock_api_client):
        """Fixture to provide a parameter manager with a mocked API client."""
        return ParameterManager(api_client=mock_api_client)

    def test_get_parameter_set_details(self, parameter_manager, mock_api_client):
        """Test retrieving details of an article."""
        mock_api_client.send_request.return_value = {"id": 1, "title": "Sample Article"}
        result = parameter_manager.get_parameter_set_details(set_id=1)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert result["id"] == 1, "The article ID should match."
        assert result["title"] == "Sample Article", "The article title should match."

    def test_list_files(self, parameter_manager, mock_api_client):
        """Test listing all files in an article."""
        mock_api_client.send_request.return_value = {
            "files": [{"name": "file1.txt"}, {"name": "file2.txt"}]
        }
        result = parameter_manager.list_files(set_id=1)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert len(result) == 2, "There should be two files."
        assert result[0]["name"] == "file1.txt", "The first file name should match."

    def test_download_files(self, parameter_manager, mock_api_client, tmp_path):
        """Test downloading files from an article."""
        mock_api_client.send_request.return_value = {
            "files": [
                {"name": "file1.txt", "download_url": "http://example.com/file1"},
                {"name": "file2.txt", "download_url": "http://example.com/file2"},
            ]
        }

        with patch(
            "Hapi.parameters.parameters.FileManager.download_file"
        ) as mock_download:
            parameter_manager.download_files(set_id=1, download_dir=tmp_path)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert mock_download.call_count == 2, "Two files should be downloaded."
        mock_download.assert_any_call(
            "http://example.com/file1", tmp_path / "file1.txt"
        )
        mock_download.assert_any_call(
            "http://example.com/file2", tmp_path / "file2.txt"
        )

    def test_get_article_id_from_friendly_id(self, parameter_manager):
        """Test mapping a friendly ID to an article ID."""
        result = parameter_manager.get_article_id(1)
        assert (
            result == ParameterManager.ARTICLE_IDS[0]
        ), "The article ID should match the corresponding friendly ID."

        result = parameter_manager.get_article_id("max")
        assert (
            result == ParameterManager.ARTICLE_IDS[-2]
        ), "The article ID should match the corresponding friendly ID."

        with pytest.raises(ValueError):
            parameter_manager.get_article_id("invalid_id")

    def test_get_article_details_with_version(self, parameter_manager, mock_api_client):
        """Test retrieving details of an article with a specific version."""
        mock_api_client.send_request.return_value = {
            "id": 1,
            "version": 2,
            "title": "Sample Article",
        }
        result = parameter_manager.get_parameter_set_details(set_id=1, version=2)

        mock_api_client.send_request.assert_called_once_with(
            "GET", "articles/19999901/versions/2"
        )
        assert result["version"] == 2, "The version should match the requested version."
        assert result["title"] == "Sample Article", "The article title should match."


@pytest.mark.integration
class TestParameterManagerIntegration:

    @pytest.fixture
    def real_api_client(self):
        """Provide an actual API client for integration testing."""
        return FigshareAPIClient()

    @pytest.fixture
    def int_test_dir(self, tmp_path):
        """Provide a temporary directory for testing file downloads."""
        return tmp_path / "integration_test_files"

    @pytest.fixture
    def parameter_manager(self, real_api_client):
        """Provide a ParameterManager with a real API client."""
        return ParameterManager(api_client=real_api_client)

    def test_integration_get_parameter_set_details(self, parameter_manager):
        """Integration test for retrieving article details."""
        set_id = 1
        article_id = 19999901
        details = parameter_manager.get_parameter_set_details(set_id)

        assert isinstance(details, dict), "Details should be a dictionary."
        assert "id" in details, "Article details should include an 'id' key."
        assert (
            details["id"] == article_id
        ), "The article ID should match the requested ID."

    def test_integration_list_files(self, parameter_manager):
        """Integration test for listing files in an article."""
        set_id = 1
        files = parameter_manager.list_files(set_id)

        assert isinstance(files, list), "Files should be returned as a list."
        assert len(files) > 0, "The article should have at least one file."
        assert "name" in files[0], "Each file should have a 'name' key."

    @pytest.mark.fig_share
    def test_integration_download_files(self, parameter_manager):
        """Integration test for downloading files from an article."""
        set_id = 1
        int_test_dir = Path("tests/rrm/data/parameters/download_files")
        int_test_dir.mkdir(parents=True, exist_ok=True)

        parameter_manager.download_files(set_id, int_test_dir)

        downloaded_files = list(int_test_dir.iterdir())
        assert (
            len(downloaded_files) == 19
        ), "Files should be downloaded to the specified directory."
        try:
            shutil.rmtree(int_test_dir)
        except PermissionError:
            pass

    def test_integration_get_article_id(self, parameter_manager):
        """Integration test for mapping a friendly ID to an article ID."""
        article_id = parameter_manager.get_article_id(1)
        assert (
            article_id == ParameterManager.ARTICLE_IDS[0]
        ), "The friendly ID should map to the correct article ID."

        article_id = parameter_manager.get_article_id("avg")
        assert (
            article_id == ParameterManager.ARTICLE_IDS[-3]
        ), "The friendly ID 'avg' should map to the correct article ID."

    def test_integration_get_article_details_with_version(self, parameter_manager):
        """Integration test for retrieving article details with a specific version."""
        set_id = 1
        version = 1
        details = parameter_manager.get_parameter_set_details(set_id, version=version)

        assert isinstance(details, dict), "Details should be a dictionary."
        assert "version" in details, "Article details should include a 'version' key."
        assert (
            details["version"] == version
        ), "The version should match the requested version."


@pytest.mark.integration
class TestParameter:

    @pytest.fixture
    def int_test_dir(self, tmp_path):
        """Provide a temporary directory for testing file downloads."""
        return tmp_path / "integration_test_parameters"

    def test_integration_get_parameters(self):
        """Integration test for downloading all parameter sets."""
        parameter = Parameter(version=1)
        int_test_dir = parameter.download_dir
        # mock to download only one parameter set
        with patch(
            "Hapi.parameters.parameters.ParameterManager.PARAMETER_SET_ID", new=[1]
        ):
            parameter.get_parameters()

        downloaded_files = list(int_test_dir.glob("**/*"))
        assert (
            len(downloaded_files) > 0
        ), "Parameter sets should be downloaded to the specified directory."

    @pytest.mark.fig_share
    def test_integration_get_parameter_set_with_download_dir(self, int_test_dir):
        """Integration test for downloading all parameter sets."""
        parameter = Parameter(version=1)
        int_test_dir.mkdir(parents=True, exist_ok=True)

        parameter.get_parameter_set(1, int_test_dir)
        downloaded_files = list(int_test_dir.glob("**/*"))

        assert (
            len(downloaded_files) > 0
        ), "Parameter sets should be downloaded to the specified directory."

    @pytest.mark.fig_share
    def test_integration_get_parameter_set_default_download_dir(self):
        """Integration test for downloading all parameter sets."""
        default_dir = os.getenv("HAPI_DATA_DIR")
        download_dir = Path(f"{default_dir}/1")

        parameter = Parameter(version=1)
        parameter.get_parameter_set(1)
        downloaded_files = list(download_dir.glob("**/*"))
        assert (
            len(downloaded_files) == 19
        ), "Parameter sets should be downloaded to the specified directory."

    def test_integration_list_parameter_names(self):
        """Integration test for listing parameter names."""
        parameter = Parameter(version=1)
        names = parameter.list_parameter_names()

        assert isinstance(names, list), "Parameter names should be returned as a list."
        assert len(names) == len(
            parameter.manager.PARAMETER_NAMES
        ), "The number of parameter names should match."
        assert (
            "01_tt" in names
        ), "Expected parameter name '01_tt' should be in the list."


@pytest.mark.mock
class TestParameterMock:

    @pytest.fixture
    def mock_parameter_manager(self):
        """Fixture to provide a mock ParameterManager."""
        return MagicMock(spec=ParameterManager)

    @pytest.fixture
    def mock_file_manager(self):
        """Fixture to provide a mock FileManager."""
        return MagicMock(spec=FileManager)

    @pytest.fixture
    def parameter(self, mock_parameter_manager):
        """Fixture to provide a Parameter instance with a mocked ParameterManager."""
        with patch("os.getenv", return_value="/mocked/path/to/data"), patch(
            "pathlib.Path.mkdir", return_value=None
        ):
            parameter_instance = Parameter(version=1)
        parameter_instance.manager = mock_parameter_manager
        return parameter_instance

    def test_get_parameters(self, parameter, mock_parameter_manager, tmp_path):
        """Test downloading all parameter sets."""

        parameter.get_parameters(tmp_path)

        # Ensure download_files was called for each parameter set ID
        assert mock_parameter_manager.download_files.call_count == len(
            ParameterManager.PARAMETER_SET_ID
        ), "download_files should be called for each parameter set ID."
        for set_id in ParameterManager.PARAMETER_SET_ID:
            mock_parameter_manager.download_files.assert_any_call(
                set_id, Path(f"{tmp_path}/{set_id}"), parameter.version
            )

    def test_get_parameter_set(self, parameter, mock_parameter_manager, tmp_path):
        """Test downloading a parameter set using a friendly ID."""
        set_id = 1
        parameter.get_parameter_set(set_id, tmp_path)

        # Ensure download_files was called with the correct arguments
        mock_parameter_manager.download_files.assert_called_once_with(
            set_id, Path(f"{tmp_path}/{set_id}"), parameter.version
        )

    def test_list_parameter_names(self):
        """Test listing parameter names."""
        names = Parameter.list_parameter_names()

        assert isinstance(names, list), "Parameter names should be returned as a list."
        assert len(names) == len(
            ParameterManager.PARAMETER_NAMES
        ), "The number of parameter names should match."
        assert (
            "01_tt" in names
        ), "Expected parameter name '01_tt' should be in the list."
