"""Hydrological model parameters.

The ``Hapi.parameters.parameters`` module provides classes for
interacting with the Figshare API to retrieve, download, and manage
global hydrological parameter sets used by the Hapi framework.

The module contains four main classes:

- ``FigshareAPIClient``: low-level HTTP client for the Figshare REST API.
- ``FileManager``: static helpers for downloading files and clearing
  directories.
- ``ParameterManager``: maps user-friendly parameter-set IDs to Figshare
  article IDs and orchestrates file listing / downloading.
- ``Parameter``: high-level facade that wires everything together and
  exposes a CLI-friendly interface.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from loguru import logger

BASE_URL = "https://api.figshare.com/v2"


class FigshareAPIClient:
    """A client for interacting with the Figshare API.

    Attributes:
        base_url (str): The base URL for the Figshare API.
        headers (dict): Headers included in every API request.

    Examples:
        >>> client = FigshareAPIClient()
    """

    def __init__(self, headers: Optional[dict] = None):
        """Initialize FigshareAPIClient.

        Args:
            headers (dict, optional): Headers to include in the API
                requests. Defaults to None, which uses
                ``{"Content-Type": "application/json"}``.
        """
        self.base_url = BASE_URL
        self.headers = headers or {"Content-Type": "application/json"}

    def send_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        binary: bool = False,
    ) -> Dict[str, int]:
        """Send an HTTP request to the Figshare API.

        Args:
            method (str): HTTP method (e.g., ``'GET'``, ``'POST'``).
            endpoint (str): API endpoint to interact with.
            data (dict, optional): Payload to include in the request.
                Defaults to None.
            binary (bool, optional): Whether the data payload is binary.
                Defaults to False.

        Returns:
            Dict[str, int]: The parsed JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Examples:
            >>> client = FigshareAPIClient()  # doctest: +SKIP
            >>> response = client.send_request(
            ...     "GET", "articles/19999901"
            ... )  # doctest: +SKIP
        """
        url = f"{self.base_url}/{endpoint}"
        payload = json.dumps(data) if data and not binary else data

        try:
            response = requests.request(method, url, headers=self.headers, data=payload)
            response.raise_for_status()
            return response.json() if response.text else None
        except requests.exceptions.HTTPError as error:
            logger.error(f"HTTPError: {error}, Response: {response.text}")
            raise

    def get_article_version(self, article_id: int, version: int) -> Dict[str, int]:
        """Retrieve a specific version of an article from the API.

        Args:
            article_id (int): The ID of the article to retrieve.
            version (int): The version number of the article to
                retrieve.

        Returns:
            Dict[str, int]: Details of the specific version of the
                article.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Examples:
            >>> client = FigshareAPIClient()  # doctest: +SKIP
            >>> response = client.get_article_version(
            ...     19999901, 1
            ... )  # doctest: +SKIP
        """
        endpoint = f"articles/{article_id}/versions/{version}"
        return self.send_request("GET", endpoint)

    def list_article_versions(self, article_id: int) -> List[Dict[str, int]]:
        """Retrieve all available versions of a specific article.

        Args:
            article_id (int): The ID of the article to retrieve
                versions for.

        Returns:
            List[Dict[str, int]]: A list of available versions for the
                specified article.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Examples:
            >>> client = FigshareAPIClient()  # doctest: +SKIP
            >>> versions = client.list_article_versions(
            ...     19999901
            ... )  # doctest: +SKIP
        """
        endpoint = f"articles/{article_id}/versions"
        return self.send_request("GET", endpoint)


class FileManager:
    r"""Handle file operations such as downloading and saving files.

    This class exposes only static methods and does not need to be
    instantiated.

    Examples:
        >>> FileManager.download_file(  # doctest: +SKIP
        ...     "https://ndownloader.figshare.com/files/35589521",
        ...     "examples/data/parameters/01_TT.tif",
        ... )
        >>> FileManager.clear_directory(
        ...     "./downloads"
        ... )  # doctest: +SKIP
    """

    @staticmethod
    def download_file(url: str, download_path: Path):
        r"""Download a file from a URL to a local path.

        Args:
            url (str): The URL of the file to download.
            download_path (Path): The local file path where the file
                will be saved.

        Raises:
            ValueError: If the URL scheme is not ``http`` or ``https``.
            requests.exceptions.HTTPError: If the download request
                fails.

        Examples:
            >>> FileManager.download_file(  # doctest: +SKIP
            ...     "https://ndownloader.figshare.com/files/35589521",
            ...     "examples/data/parameters/01_TT.tif",
            ... )
        """
        allowed_schemes = {"http", "https"}
        scheme = urlparse(url).scheme
        if scheme not in allowed_schemes:
            raise ValueError(f"URL scheme '{scheme}' is not allowed.")

        download_path = (
            Path(download_path) if isinstance(download_path, str) else download_path
        )
        download_path.parent.mkdir(parents=True, exist_ok=True)

        # Perform the download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.debug(f"File downloaded: {download_path}")

    @staticmethod
    def clear_directory(directory: Union[Path, str]):
        """Clear all files in the specified directory.

        Only regular files are removed; subdirectories are left
        untouched.

        Args:
            directory (Union[Path, str]): The directory to clear.

        Examples:
            >>> FileManager.clear_directory(
            ...     "./downloads"
            ... )  # doctest: +SKIP
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        if directory.exists():
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()
            logger.debug(f"Cleared directory: {directory}")


class ParameterManager:
    r"""Manage hydrological parameters via the Figshare API.

    This class maps user-friendly parameter-set identifiers (1--10,
    ``"avg"``, ``"max"``, ``"min"``) to Figshare article IDs and
    provides convenience methods for listing and downloading files.

    Attributes:
        ARTICLE_IDS (list): Figshare article IDs for each parameter
            set.
        PARAMETER_NAMES (list): Canonical names of all 18 HBV
            parameters.
        PARAMETER_SET_ID (list): User-friendly IDs (1--10, ``"avg"``,
            ``"max"``, ``"min"``).
        api_client (FigshareAPIClient): The underlying API client.

    Examples:
        >>> api_client = FigshareAPIClient()
        >>> manager = ParameterManager(api_client)
        >>> files = manager.list_files(1)  # doctest: +SKIP
        >>> details = manager.get_parameter_set_details(
        ...     1
        ... )  # doctest: +SKIP
        >>> manager.download_files(
        ...     1, "examples/data/downloads"
        ... )  # doctest: +SKIP
    """

    ARTICLE_IDS = [
        19999901,
        19999988,
        19999997,
        20000006,
        20000012,
        20000018,
        20000015,
        20000024,
        20000027,
        20000030,
        20153402,
        20153405,
        20362374,
    ]

    PARAMETER_NAMES = [
        "01_tt",
        "02_rfcf",
        "03_sfcf",
        "04_cfmax",
        "05_cwh",
        "06_cfr",
        "07_fc",
        "08_beta",
        "09_etf",
        "10_lp",
        "11_k0",
        "12_k1",
        "13_k2",
        "14_uzl",
        "15_perc",
        "16_maxbas",
        "17_K_muskingum",
        "18_x_muskingum",
    ]

    PARAMETER_SET_ID = list(range(1, 11)) + ["avg", "max", "min"]

    def __init__(self, api_client: FigshareAPIClient):
        """Initialize ParameterManager.

        Args:
            api_client (FigshareAPIClient): A configured Figshare API
                client instance.
        """
        self.api_client = api_client

    def get_parameter_set_details(
        self, set_id: Union[int, str], version: Optional[int] = None
    ) -> Dict[str, int]:
        """Retrieve details of a parameter set from the Figshare API.

        Args:
            set_id (Union[int, str]): The user-friendly ID of the
                parameter set (1--10, ``"avg"``, ``"max"``, ``"min"``).
            version (int, optional): The version of the parameter set.
                Defaults to None (latest version).

        Returns:
            Dict[str, int]: A dictionary containing full article
                metadata (files, authors, license, dates, etc.).

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If *set_id* is not a valid parameter-set ID.

        Examples:
            >>> api_client = FigshareAPIClient()
            >>> manager = ParameterManager(api_client)
            >>> details = manager.get_parameter_set_details(
            ...     1
            ... )  # doctest: +SKIP
        """
        article_id = self.get_article_id(set_id)
        endpoint = f"articles/{article_id}"
        if version:
            endpoint += f"/versions/{version}"
        return self.api_client.send_request("GET", endpoint)

    def list_files(self, set_id: Union[int, str], version: Optional[int] = None):
        """List all files in a parameter set.

        Args:
            set_id (Union[int, str]): The user-friendly ID of the
                parameter set.
            version (int, optional): The version of the article.
                Defaults to None.

        Returns:
            list: A list of file dictionaries, each containing keys
                such as ``"id"``, ``"name"``, and
                ``"download_url"``.

        Examples:
            >>> api_client = FigshareAPIClient()
            >>> manager = ParameterManager(api_client)
            >>> files = manager.list_files(1)  # doctest: +SKIP
        """
        details = self.get_parameter_set_details(set_id, version)
        return details.get("files", [])

    def download_files(
        self, set_id: Union[int, str], download_dir: Path, version: Optional[int] = None
    ):
        r"""Download all files in a parameter set to a local directory.

        Args:
            set_id (Union[int, str]): The user-friendly ID of the
                parameter set.
            download_dir (Path): The local directory to save the files.
            version (int, optional): The version of the article.
                Defaults to None.

        Examples:
            >>> api_client = FigshareAPIClient()
            >>> manager = ParameterManager(api_client)
            >>> manager.download_files(
            ...     1, "examples/data/downloads"
            ... )  # doctest: +SKIP
        """
        download_dir = (
            Path(download_dir) if isinstance(download_dir, str) else download_dir
        )
        files = self.list_files(set_id, version)

        for file in files:
            dest_path = download_dir / file["name"]
            FileManager.download_file(file["download_url"], dest_path)

    def get_article_id(self, set_id: Union[int, str]) -> int:
        """Map a user-friendly set ID to a Figshare article ID.

        Args:
            set_id (Union[int, str]): The parameter set ID (1--10,
                ``"avg"``, ``"max"``, ``"min"``).

        Returns:
            int: The corresponding Figshare article ID.

        Raises:
            ValueError: If *set_id* is not found in
                ``PARAMETER_SET_ID``.

        Examples:
            >>> api_client = FigshareAPIClient()
            >>> manager = ParameterManager(api_client)
            >>> manager.get_article_id(1)
            19999901
        """
        try:
            index = self.PARAMETER_SET_ID.index(set_id)
            return self.ARTICLE_IDS[index]
        except ValueError:
            raise ValueError(
                f"Invalid Parameter Set ID: {set_id}, valid IDs: {self.PARAMETER_SET_ID}"
            )


class Parameter:
    r"""High-level interface for handling hydrological parameters.

    The ``HAPI_DATA_DIR`` environment variable must be set to the
    directory where parameter sets will be saved when *download_dir* is
    not provided.

    Attributes:
        version (int): The version of the parameter sets to retrieve.
        api_client (FigshareAPIClient): The underlying Figshare API
            client.
        manager (ParameterManager): The parameter manager that
            handles article lookups and downloads.
        download_dir (Path): The directory where parameter sets are
            saved.

    Examples:
        >>> parameter = Parameter(version=1)  # doctest: +SKIP
        >>> parameter.get_parameters(
        ...     "examples/data/parameters"
        ... )  # doctest: +SKIP
        >>> parameter.get_parameter_set(
        ...     1, "examples/data/parameters"
        ... )  # doctest: +SKIP
        >>> names = Parameter.list_parameter_names()
        >>> len(names)
        18
        >>> names[0]
        '01_tt'
        >>> names[-1]
        '18_x_muskingum'
    """

    def __init__(self, version: int = 1, download_dir: Optional[Path] = None):
        """Initialize Parameter.

        Args:
            version (int, optional): The Figshare article version to
                use. Defaults to 1.
            download_dir (Path, optional): Directory where parameter
                files are saved. Defaults to None, which reads from
                the ``HAPI_DATA_DIR`` environment variable.

        Raises:
            ValueError: If *download_dir* is None and the
                ``HAPI_DATA_DIR`` environment variable is not set.
        """
        self.version = version
        self.api_client = FigshareAPIClient()
        self.manager = ParameterManager(self.api_client)
        if download_dir is None:
            download_dir = os.getenv("HAPI_DATA_DIR")
            if download_dir is None:
                raise ValueError("HAPI_DATA_DIR environment variable is not set")
            else:
                download_dir = Path(download_dir)
                download_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir = download_dir

    def get_parameters(self, download_dir: Optional[Path] = None):
        r"""Download all parameter sets.

        Iterates over every entry in
        ``ParameterManager.PARAMETER_SET_ID`` and downloads the
        corresponding files.

        Args:
            download_dir (Path, optional): The directory where
                parameter sets will be saved. Defaults to None (uses
                the instance ``download_dir``).

        Examples:
            >>> parameter = Parameter(version=1)  # doctest: +SKIP
            >>> parameter.get_parameters(
            ...     "examples/data/parameters"
            ... )  # doctest: +SKIP
        """
        for set_id in ParameterManager.PARAMETER_SET_ID:
            self.get_parameter_set(set_id, download_dir)
            logger.debug(f"Downloaded parameter set: {set_id} to {download_dir}")

    def get_parameter_set(
        self, set_id: Union[int, str], download_dir: Optional[Path] = None
    ):
        r"""Download a specific parameter set.

        Args:
            set_id (Union[int, str]): The user-friendly ID of the
                parameter set to download (1--10, ``"avg"``, ``"max"``,
                ``"min"``).
            download_dir (Path, optional): The directory where the
                parameter set will be saved. Defaults to None (uses
                the instance ``download_dir`` with a subdirectory
                named after *set_id*).

        Raises:
            ValueError: If *set_id* is not a valid parameter-set ID.

        Examples:
            >>> parameter = Parameter(version=1)  # doctest: +SKIP
            >>> parameter.get_parameter_set(
            ...     1, "examples/data/parameters"
            ... )  # doctest: +SKIP
        """
        if set_id not in ParameterManager.PARAMETER_SET_ID:
            raise ValueError(
                f"Invalid friendly ID: {set_id}, valid IDs: {ParameterManager.PARAMETER_SET_ID}"
            )

        if download_dir is None:
            download_dir = self.download_dir / f"{set_id}"
        else:
            download_dir = Path(download_dir) / f"{set_id}"

        self.manager.download_files(set_id, download_dir, self.version)
        logger.debug(f"Downloaded parameter set: {set_id} to {download_dir}")

    @staticmethod
    def list_parameter_names() -> List[str]:
        """List all parameter names.

        Returns:
            List[str]: A list of the 18 HBV parameter names.

        Examples:
            >>> names = Parameter.list_parameter_names()
            >>> len(names)
            18
            >>> names[0]
            '01_tt'
            >>> names[-1]
            '18_x_muskingum'
        """
        return ParameterManager.PARAMETER_NAMES


def main():
    """Run the Hapi CLI for hydrological parameter operations.

    This entry point provides three sub-commands:

    - ``download-parameters`` -- download every parameter set.
    - ``download-parameter-set <set_id>`` -- download one parameter
      set.
    - ``list-parameter-names`` -- print all parameter names to stdout.

    Raises:
        ValueError: If invalid or insufficient arguments are provided.
        FileNotFoundError: If the specified directory does not exist
            and cannot be created.
        requests.exceptions.RequestException: If there is an error
            communicating with the Figshare API.

    Examples:
        Download all parameter sets::

            download-parameters --directory /path/to/save --version 1

        Download a specific parameter set::

            download-parameter-set 1 --directory /path/to/save \\
                --version 1

        List parameter names::

            list-parameter-names

    See Also:
        ``Hapi.parameters.parameters.Parameter``: For details on the
            ``Parameter`` class and its methods.
        ``Hapi.parameters.parameters.ParameterManager``: For managing
            parameter-related operations.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Hapi CLI for parameter operations.")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: download-parameters
    download_params_parser = subparsers.add_parser(
        "download-parameters", help="Download all parameter sets."
    )
    download_params_parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory to save downloaded parameters. Defaults to HAPI_DATA_DIR.",
    )
    download_params_parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version of the parameter sets to download. Defaults to 1.",
    )

    # Command: download-parameter-set
    download_param_set_parser = subparsers.add_parser(
        "download-parameter-set", help="Download a specific parameter set."
    )
    download_param_set_parser.add_argument(
        "set_id",
        type=str,
        help="ID of the parameter set to download (e.g., 1, avg, max).",
    )
    download_param_set_parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory to save downloaded parameter set. Defaults to HAPI_DATA_DIR.",
    )
    download_param_set_parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version of the parameter set to download. Defaults to 1.",
    )

    # Command: list-parameter-names
    subparsers.add_parser(
        "list-parameter-names", help="List all available parameter names."
    )

    args = parser.parse_args()

    if args.command == "download-parameters":
        parameter = Parameter(version=args.version)
        parameter.get_parameters(download_dir=args.directory)

    elif args.command == "download-parameter-set":
        parameter = Parameter(version=args.version)
        parameter.get_parameter_set(set_id=args.set_id, download_dir=args.directory)

    elif args.command == "list-parameter-names":
        names = Parameter.list_parameter_names()
        print("Available parameter names:")
        for name in names:
            print(f"- {name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
