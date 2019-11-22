import os
import csv
import multiprocessing
from typing import Iterable, List, Optional, Tuple

import click
import numpy as np
import rasterio
from parallelpipe import stage

WORKERS: int = multiprocessing.cpu_count()
QSIZE: int = 5


@click.command()
@click.argument("tiles", type=str)
@click.option(
    "-m", "--method", default="linear", type=str, help="Method for creating bins"
)
@click.option(
    "-w", "--workers", default=None, type=int, help="Number of parallel workers"
)
@click.option(
    "--minmax_only",
    is_flag=True,
    default=False,
    type=bool,
    help="Only compute minmax, not histogram",
)
def cli(tiles: str, method: str, workers: Optional[int], minmax_only: bool):

    histo_range: Tuple[int, int]
    bins: int
    offset: float
    min_value: float
    max_value: float

    if workers:
        global WORKERS
        WORKERS = workers

    # sources: List[str] = get_tiles(tiles)
    sources = ["s3://gfw-files/tmaschler/bio-intact/0000093184-0000093184.tif"]
    click.echo("Processing sources:")
    click.echo(sources)

    min_value, max_value = compute_min_max(sources)

    if not minmax_only:
        histo_range, bins, offset = get_range(min_value, max_value, method)
        get_histo(sources, histo_range, bins, offset, method)


def get_tiles(f: str):

    tiles: List[str] = list()

    with open(os.path.join(f)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            tiles.append(row[0])

    return tiles


def get_histo(
    sources: List[str],
    histo_range: Tuple[int, int],
    bins: int,
    offset: float,
    method: str,
) -> None:
    pipe = sources | process_sources(histo_range, bins, method)

    first: bool = True
    result: Optional[np.ndarray] = None

    for histo in pipe.results():
        if first:
            result = histo[0]
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


def compute_min_max(sources: List[str]) -> Tuple[float, float]:
    min_value: float = 0
    max_value: float = 0

    pipe = sources | get_min_max
    for minmax in pipe.results():
        if minmax[0] < min_value:
            min_value = minmax[0]
        if minmax[1] > max_value:
            max_value = minmax[1]

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
        if min_value < 0:
            offset = abs(min_value) + 1
        histo_range = (
            np.log(min_value + offset) * 1000,
            np.log(max_value + offset) * 1000,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    bins: int = len(range(histo_range[0], histo_range[1]))
    click.echo(f"RANGE: {histo_range}, BINS: {bins}, OFFSET: {offset}")

    return histo_range, bins, offset


@stage(workers=WORKERS, qsize=QSIZE)
def process_sources(
    sources: Iterable[str], histo_range: Tuple[int, int], bins: int, method: str
) -> Iterable[Tuple[np.ndarray]]:
    """
    Loops over all blocks and reads first input raster to get coordinates.
    Append values from all input rasters
    """

    for source in sources:
        click.echo(source)

        try:
            with rasterio.open(source) as src:
                w: np.ndarray = src.read(1)
        except rasterio.RasterioIOError:
            click.echo(f"Could not read {source}")
            raise

        w = w[~np.isnan(w)]

        if method == "linear":
            w = (w * 100).astype(np.int16)
        elif method == "log":
            w = (np.log(w + 100) * 1000).astype(np.int16)
        else:
            raise ValueError(f"Unknown method {method}")

        histo: np.ndarray = _compute_histogram(w, bins, histo_range)
        w = None
        yield (histo,)


@stage(workers=WORKERS, qsize=QSIZE)
def get_min_max(sources: Iterable[str]) -> Iterable[Tuple[int, int]]:

    for source in sources:
        click.echo(f"Reading tile {source}")

        with rasterio.open(source) as src:
            w = src.read(1)

        w = w[~np.isnan(w)]

        if len(w):
            min = np.amin(w)
            max = np.amax(w)
            w = None
            click.echo(f"Tile {source} has min {min} and max {max}")
            yield (min, max)
        else:
            click.echo(f"Tile {source} is empty")


def _add_histogram(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return h1 + h2


def _compute_histogram(w: np.ndarray, bins: int, range: Tuple[int, int]) -> np.ndarray:
    return np.histogram(w, bins, range)[0]


if __name__ == "__main__":
    cli()
