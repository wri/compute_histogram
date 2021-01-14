import json
import os
import csv

from typing import List, Optional, Tuple, Dict, Any
from multiprocessing import Pool
from multiprocessing import cpu_count
from urllib.parse import urlparse

import boto3
import click
import numpy as np
import rasterio
from itertools import repeat
from rasterio import RasterioIOError
from retrying import retry


@click.command()
@click.argument("source", type=str)
@click.option(
    "-m", "--method", default="linear", type=str, help="Method for creating bins"
)
@click.option(
    "-w", "--workers", default=cpu_count(), type=int, help="Number of parallel workers"
)
@click.option("--min_value", default=None, type=float, help="Minimum value in tile set")
@click.option("--max_value", default=None, type=float, help="Maximum value in tile set")
@click.option(
    "--minmax_only",
    is_flag=True,
    default=False,
    type=bool,
    help="Only compute minmax, not histogram",
)
def cli(
    source: str,
    method: str,
    workers: int,
    min_value: Optional[float],
    max_value: Optional[float],
    minmax_only: bool,
):
    histo_range: Tuple[int, int]
    bins: int
    offset: float

    sources: List[str] = get_tiles(source)
    # sources = ["s3://gfw-files/tmaschler/bio-intact/0000093184-0000093184.tif"]
    click.echo("Processing sources:")
    click.echo(sources)

    if not (min_value and max_value):
        min_value, max_value = compute_min_max(sources, workers)

    histo_range, bins, offset = get_range(min_value, max_value, method)
    get_histo(sources, histo_range, bins, offset, method, workers)


def get_tiles(f: str):

    bucket, key = split_s3_path(f)

    s3_client = boto3.client("s3")
    tiles_resp = s3_client.get_object(Bucket=bucket, Key=key)
    tiles_geojson: Dict[str, Any] = json.loads(
        tiles_resp["Body"].read().decode("utf-8")
    )

    tiles: List[str] = list()
    for feature in tiles_geojson["features"]:
        tiles.append(feature["properties"]["name"])

    return tiles


def split_s3_path(s3_path: str) -> Tuple[str, str]:
    o = urlparse(s3_path, allow_fragments=False)
    return o.netloc, o.path.lstrip("/")


def get_histo(
    sources: List[str],
    histo_range: Tuple[int, int],
    bins: int,
    offset: float,
    method: str,
    workers: int,
) -> None:
    pool = Pool(workers)
    results = pool.starmap(
        process_sources, zip(sources, repeat(histo_range), repeat(bins), repeat(method))
    )
    pool.close()  # 'TERM'
    pool.join()  # 'KILL'

    first: bool = True
    result: Optional[np.ndarray] = None

    for histo in results:
        if first:
            result = histo
            first = False
        else:
            result = _add_histogram(result, histo)

    if method == "linear":
        bins = np.array([x / 100 for x in range(histo_range[0], histo_range[1])])
    elif method == "log":
        bins = np.array(
            [np.exp(x / 1000) - offset for x in range(histo_range[0], histo_range[1])]
        )
    else:
        raise ValueError(f"Unknown method {method}")

    histogram: np.ndarray = np.vstack((bins, result)).T

    click.echo("Histogram: ")
    click.echo(histogram)

    np.savetxt("histogram.csv", histogram, fmt="%1.2f, %d")


def compute_min_max(sources: List[str], workers: int) -> Tuple[float, float]:
    min_value: float = 0
    max_value: float = 0

    pool = Pool(workers)
    results = pool.map(get_min_max, sources)
    pool.close()  # 'TERM'
    pool.join()  # 'KILL'

    for result in results:
        if result:
            if result[0] < min_value:
                min_value = result[0]
            if result[1] > max_value:
                max_value = result[1]
            click.echo(f"MIN: {min_value}, MAX: {max_value}")

    click.echo("DONE MIN MAX. Final result:")
    click.echo(f"MIN: {min_value}, MAX: {max_value}")
    return min_value, max_value


def get_range(
    min_value: float, max_value: float, method: str
) -> Tuple[Tuple[int, int], int, float]:
    offset: float = 0

    click.echo(f"Min: {min_value}, Max: {max_value}")

    if method == "linear":
        histo_range: Tuple[int, int] = (
            int(min_value * 100) - 10,
            int(max_value * 100) + 10,
        )
    elif method == "log":
        if min_value <= 0:
            offset = abs(min_value) + 1
        histo_range = (
            int(np.log(min_value + offset) * 1000),
            int(np.log(max_value + offset) * 1000),
        )
    else:
        raise ValueError(f"Unknown method {method}")

    bins: int = len(range(histo_range[0], histo_range[1]))
    click.echo(f"RANGE: {histo_range}, BINS: {bins}, OFFSET: {offset}")

    return histo_range, bins, offset


def process_sources(
    source: str, histo_range: Tuple[int, int], bins: int, method: str
) -> np.ndarray:
    """
    Loops over all blocks and reads first input raster to get coordinates.
    Append values from all input rasters
    """

    click.echo(source)

    w = read_source(source)

    if method == "linear":
        w = (w * 100).astype(np.int16)
    elif method == "log":
        w = (np.log(w + 100) * 1000).astype(np.int16)
    else:
        raise ValueError(f"Unknown method {method}")

    histo: np.ndarray = _compute_histogram(w, bins, histo_range)
    w = None
    return histo


def get_min_max(source: str) -> Optional[Tuple[int, int]]:
    click.echo(f"Reading tile {source}")

    w = read_source(source)

    if len(w):
        min = np.amin(w)
        max = np.amax(w)
        w = None
        click.echo(f"Tile {source} has min {min} and max {max}")
        return min, max
    else:
        click.echo(f"Tile {source} is empty")
        return None


def retry_if_rasterio_io_error(exception) -> bool:
    is_rasterio_io_error: bool = isinstance(
        exception, RasterioIOError
    ) and "IReadBlock failed" in str(exception)
    if is_rasterio_io_error:
        print("RasterioIO Error - RETRY")
    return is_rasterio_io_error


@retry(
    retry_on_exception=retry_if_rasterio_io_error,
    stop_max_attempt_number=7,
    wait_exponential_multiplier=1000,
    wait_exponential_max=300000,
)
def read_source(source):
    with rasterio.open(source) as src:
        w = src.read(1)

    return w[~np.isnan(w)]


def _add_histogram(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return h1 + h2


def _compute_histogram(w: np.ndarray, bins: int, range: Tuple[int, int]) -> np.ndarray:
    return np.histogram(w, bins, range)[0]


if __name__ == "__main__":
    cli()
