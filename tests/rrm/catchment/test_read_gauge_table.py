"""Unit tests for ``Catchment.read_gauge_table``, covering both input formats.

The CSV branch is exercised by ``tests/rrm/catchment/test_rrm_catchment.py``; the
GeoJSON branch had no coverage at all, which is why it is the focus here. That branch is
where vector reading moved from ``geopandas.read_file`` to
``pyramids.feature.FeatureCollection.read_file``, so these tests pin the behaviour the
swap has to preserve: the same rows, the same columns, and a frame that still behaves as
a ``GeoDataFrame`` for everything downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pyramids.feature import FeatureCollection
from shapely.geometry import Point

from hapi.catchment import Catchment

GAUGES = [
    (1, "Station 1", 454795.6728, 503143.3264),
    (2, "Station 2", 443847.5736, 481850.7151),
]
"""Two Coello gauges (id, name, easting, northing) in the raster's UTM 18N frame."""


@pytest.fixture(scope="function")
def gauges_geojson(tmp_path) -> str:
    """Write a two-gauge GeoJSON in the Coello CRS and return its path.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        str: Path to the written ``.geojson`` file.
    """
    gdf = FeatureCollection(
        {
            "id": [row[0] for row in GAUGES],
            "name": [row[1] for row in GAUGES],
        },
        geometry=[Point(row[2], row[3]) for row in GAUGES],
        crs="EPSG:32618",
    )
    path = tmp_path / "gauges.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture(scope="function")
def catchment(coello_start_date: str, coello_end_date: str) -> Catchment:
    """Return a distributed Catchment spanning the Coello test period.

    Args:
        coello_start_date: Start date fixture.
        coello_end_date: End date fixture.

    Returns:
        Catchment: An instance with no inputs read yet.
    """
    return Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
        fmt="%Y-%m-%d",
    )


class TestReadGaugeTable:
    """Tests for ``Catchment.read_gauge_table``."""

    def test_geojson_is_read_as_a_geodataframe(self, catchment, gauges_geojson):
        """Test that a GeoJSON gauge file loads with its geometry intact.

        Test scenario:
            The GeoJSON branch must produce something that still satisfies the
            ``GeoDataFrame`` contract every downstream consumer relies on — ``.loc``,
            ``.columns``, ``len()`` and an active geometry column — regardless of which
            library performed the read.
        """
        catchment.read_gauge_table(gauges_geojson)

        assert isinstance(catchment.GaugesTable, FeatureCollection), (
            "the GeoJSON branch must yield a FeatureCollection (a GeoDataFrame subclass), "
            f"got {type(catchment.GaugesTable).__name__}"
        )
        assert len(catchment.GaugesTable) == len(GAUGES), (
            f"expected {len(GAUGES)} gauges, got {len(catchment.GaugesTable)}"
        )
        assert catchment.GaugesTable["id"].tolist() == [1, 2], (
            f"gauge ids should survive the read, got {catchment.GaugesTable['id'].tolist()}"
        )
        assert catchment.GaugesTable.geometry.iloc[0].x == pytest.approx(
            GAUGES[0][2]
        ), (
            "the first gauge's easting should survive the read, got "
            f"{catchment.GaugesTable.geometry.iloc[0].x}"
        )

    def test_geojson_preserves_the_crs(self, catchment, gauges_geojson):
        """Test that the declared CRS survives the read.

        Test scenario:
            Gauge coordinates are matched against the flow-accumulation grid, so losing
            the CRS would silently misplace every station.
        """
        catchment.read_gauge_table(gauges_geojson)

        assert catchment.GaugesTable.crs is not None, "the gauge table lost its CRS"
        assert catchment.GaugesTable.crs.to_epsg() == 32618, (
            f"expected EPSG:32618, got {catchment.GaugesTable.crs.to_epsg()}"
        )

    def test_geojson_maps_gauges_onto_the_grid(
        self, catchment, gauges_geojson, coello_acc_path
    ):
        """Test that gauge coordinates resolve to cell indices via the flow-acc raster.

        Args:
            coello_acc_path: Path fixture for the Coello accumulation raster.

        Test scenario:
            With a ``flow_acc_file`` given and no ``cell_row`` column present, the reader
            asks pyramids to map each gauge onto the grid. This is the step that consumes
            the frame produced by the vector read, so it is the real compatibility check
            between the reader and ``Dataset.map_to_array_coordinates``.
        """
        catchment.read_gauge_table(gauges_geojson, coello_acc_path)

        for column in ("cell_row", "cell_col"):
            assert column in catchment.GaugesTable.columns, (
                f"expected a {column} column after mapping, got "
                f"{catchment.GaugesTable.columns.tolist()}"
            )
        indices = catchment.GaugesTable[["cell_row", "cell_col"]].to_numpy()
        assert np.issubdtype(indices.dtype, np.number), (
            f"cell indices should be numeric, got dtype {indices.dtype}"
        )
        assert (indices >= 0).all(), f"cell indices must be non-negative, got {indices}"

    def test_csv_branch_is_a_plain_dataframe(self, catchment, coello_gauges_table):
        """Test that the CSV branch still returns a non-spatial DataFrame.

        Args:
            coello_gauges_table: Path fixture for the Coello gauge CSV.

        Test scenario:
            Only the GeoJSON branch changed. A CSV has no geometry, so it must keep
            loading as a plain ``DataFrame`` and must *not* be promoted to a
            ``GeoDataFrame``.
        """
        catchment.read_gauge_table(coello_gauges_table)

        assert isinstance(catchment.GaugesTable, DataFrame), (
            f"expected a DataFrame, got {type(catchment.GaugesTable).__name__}"
        )
        assert not hasattr(catchment.GaugesTable, "crs"), (
            "a CSV gauge table has no geometry and must not become a spatial frame"
        )

    def test_start_and_end_columns_are_parsed_as_dates(self, catchment, tmp_path):
        """Test that ``start``/``end`` columns are converted to datetimes.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            When the table carries a validity period, both columns are parsed with the
            supplied format. Covers the CSV branch, where the columns are strings on read.
        """
        path = tmp_path / "gauges_dates.csv"
        pd.DataFrame(
            {
                "id": [1],
                "name": ["Station 1"],
                "start": ["2009-01-01"],
                "end": ["2011-12-31"],
            }
        ).to_csv(path, index=False)

        catchment.read_gauge_table(str(path))

        assert catchment.GaugesTable.loc[0, "start"].year == 2009, (
            f"start should parse to 2009, got {catchment.GaugesTable.loc[0, 'start']}"
        )
        assert catchment.GaugesTable.loc[0, "end"].year == 2011, (
            f"end should parse to 2011, got {catchment.GaugesTable.loc[0, 'end']}"
        )

    def test_start_and_end_become_datetime_columns(self, catchment, tmp_path):
        """Test that the parsed columns carry a datetime dtype, not object.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The whole column is converted at once, so the result is a real
            ``datetime64`` column rather than a column of Python objects. Downstream
            comparisons against the model's ``Index`` rely on that.
        """
        path = tmp_path / "gauges_dtype.csv"
        pd.DataFrame(
            {
                "id": [1, 2],
                "start": ["2009-01-01", "2009-02-01"],
                "end": ["2011-12-31", "2011-11-30"],
            }
        ).to_csv(path, index=False)

        catchment.read_gauge_table(str(path))

        for column in ("start", "end"):
            assert pd.api.types.is_datetime64_any_dtype(
                catchment.GaugesTable[column]
            ), (
                f"{column} should be a datetime column, got dtype "
                f"{catchment.GaugesTable[column].dtype}"
            )

    def test_start_without_end_is_parsed(self, catchment, tmp_path):
        """Test that a table carrying only ``start`` is handled.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The columns are now converted independently. Previously a single ``start in
            columns`` check guarded both conversions, so a table with ``start`` but no
            ``end`` raised ``KeyError`` on the missing column.
        """
        path = tmp_path / "gauges_start_only.csv"
        pd.DataFrame({"id": [1], "start": ["2009-01-01"]}).to_csv(path, index=False)

        catchment.read_gauge_table(str(path))

        assert catchment.GaugesTable.loc[0, "start"].year == 2009, (
            f"start should parse to 2009, got {catchment.GaugesTable.loc[0, 'start']}"
        )
        assert "end" not in catchment.GaugesTable.columns, (
            "no end column should be invented when the source has none"
        )

    def test_custom_date_format_is_honoured(self, catchment, tmp_path):
        """Test that ``fmt`` drives the parsing.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            A day-first table parsed with ``"%d/%m/%Y"`` must read 03/04/2009 as 3 April,
            not 4 March — proving the format is applied rather than guessed.
        """
        path = tmp_path / "gauges_dayfirst.csv"
        pd.DataFrame(
            {"id": [1], "start": ["03/04/2009"], "end": ["05/06/2011"]}
        ).to_csv(path, index=False)

        catchment.read_gauge_table(str(path), fmt="%d/%m/%Y")

        parsed = catchment.GaugesTable.loc[0, "start"]
        assert (parsed.day, parsed.month, parsed.year) == (3, 4, 2009), (
            f"expected 3 April 2009 under a day-first format, got {parsed}"
        )

    def test_blank_date_raises(self, catchment, tmp_path):
        """Test that a blank ``start``/``end`` cell is rejected rather than becoming NaT.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            ``pd.to_datetime`` maps a blank cell to ``NaT`` instead of raising, where the
            per-cell ``strptime`` it replaced rejected it. A gauge with no validity period
            is almost always a data-entry slip, and a silent ``NaT`` would flow into the
            period comparisons unnoticed.
        """
        path = tmp_path / "gauges_blank.csv"
        pd.DataFrame(
            {
                "id": [1, 2],
                "start": ["2009-01-01", ""],
                "end": ["2011-12-31", "2011-12-31"],
            }
        ).to_csv(path, index=False)

        with pytest.raises(ValueError, match="no usable date"):
            catchment.read_gauge_table(str(path))

    def test_date_not_matching_the_format_raises(self, catchment, tmp_path):
        """Test that an unparseable date is rejected.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            A value that does not match ``fmt`` raises ``ValueError``, the same failure
            mode the previous per-cell ``datetime.strptime`` produced.
        """
        path = tmp_path / "gauges_bad.csv"
        pd.DataFrame(
            {"id": [1], "start": ["not-a-date"], "end": ["2011-12-31"]}
        ).to_csv(path, index=False)

        with pytest.raises(ValueError):
            catchment.read_gauge_table(str(path))

    def test_geojson_dates_are_parsed_too(self, catchment, tmp_path):
        """Test that the date conversion also applies on the GeoJSON branch.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The conversion runs after the format branch, so it must work on the
            ``FeatureCollection`` a GeoJSON produces as well as on a plain DataFrame —
            assigning a datetime column onto a GeoDataFrame must not disturb its geometry.
        """
        gdf = FeatureCollection(
            {"id": [1], "start": ["2009-01-01"], "end": ["2011-12-31"]},
            geometry=[Point(GAUGES[0][2], GAUGES[0][3])],
            crs="EPSG:32618",
        )
        path = tmp_path / "gauges_dates.geojson"
        gdf.to_file(path, driver="GeoJSON")

        catchment.read_gauge_table(str(path))

        assert catchment.GaugesTable.loc[0, "start"].year == 2009, (
            f"start should parse to 2009, got {catchment.GaugesTable.loc[0, 'start']}"
        )
        assert catchment.GaugesTable.geometry.iloc[0].x == pytest.approx(
            GAUGES[0][2]
        ), "the geometry should survive the date conversion"
