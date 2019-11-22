from parallelpipe import stage
import rasterio
import numpy as np
import os
import csv
import subprocess as sp


HISTO_RANGE = (-2000, 2001)
# HISTO_RANGE = (0, 1651)
BINS = len(range(HISTO_RANGE[0], HISTO_RANGE[1]))
MAX_BLOCK_SIZE = 4000
WORKERS = 25
QSIZE = 5

PATHS = [
    "/vsis3/gfw-files/2018_update/biodiversity_significance/{tile_id}.tif",
    "/vsis3/gfw-files/2018_update/plantations/{tile_id}.tif",
]
TILE_CSV = "csv/bio_sig_list.txt"


@stage(workers=WORKERS, qsize=QSIZE)
def process_sources(
    sources
):
    """
    Loops over all blocks and reads first input raster to get coordinates.
    Append values from all input rasters
    :param blocks: list of blocks to process
    :param src_rasters: list of input rasters
    :param col_size: pixel width
    :param row_size: pixel height
    :param step_width: block width
    :param step_height: block height
    :param width: image width
    :param height: image height
    :return: Table of Lat/Lon coord and corresponding raster values
    """

    for source in sources:
        print(source)

        with rasterio.open(source) as src1:
            w = (src1.read(1) * 100).astype(np.uint16)
        w_m = _apply_mask(_get_mask(w, 0), w)
        histo = _compute_histogram(w_m, BINS, HISTO_RANGE)
        w_m = None
        yield (histo, )


@stage(workers=WORKERS, qsize=QSIZE)
def get_min_max(sources):

    for source in sources:
        print(source)

        with rasterio.open(source) as src1:
            w = (src1.read(1))

        min = np.amin(w)
        max = np.amax(w)
        w = None
        yield (min, max)


def get_tiles():

    tiles = list()

    dir = os.path.dirname(__file__)
    with open(os.path.join(dir, TILE_CSV)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            tiles.append(row[0])

    return tiles


def add_histogram(h1, h2):
    return h1 + h2


def _get_mask(w, no_data=0, mask_w=None):
    if mask_w is not None:
        return mask_w * (w != no_data)
    else:
        return w != no_data


def _apply_mask(mask, w):
    return np.extract(mask, w)


def _compute_histogram(w, bins, range):
    return np.histogram(w, bins, range)[0]


def _get_top(coord):
    if coord >= 0:
        return f"{coord:02}" + "N"
    else:
        return f"{-coord:02}" + "S"


def _get_left(coord):
    if coord >= 0:
        return f"{coord:003}" + "E"
    else:
        return f"{-coord:003}" + "W"


def get_histo(sources):
    pipe = sources | process_sources

    first = True
    result = None
    result_m = None
    for histo in pipe.results():
        if first:
            result = histo[0]
            first = False
        else:
            result = add_histogram(result, histo)

    bins = np.exp(np.array([x / 100 for x in range(HISTO_RANGE[0], HISTO_RANGE[1])]))
    histogram = np.vstack((bins, result)).T

    print("Histogram: ")
    print(histogram)

    np.savetxt("histogram.csv", histogram, fmt='%1.2f, %d')

@stage(workers=WORKERS, qsize=QSIZE)
def warp(sources):
    for source in sources:
        local_src = os.path.join("/mnt/data/img/sig", os.path.basename(source))
        cmd = ["gdalwarp", "-co", "COMPRESS=LZW", source, local_src]
        p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        o, e = p.communicate()
        print(o)
        if p.returncode != 0:
            raise Exception(e)
        yield local_src


if __name__ == "__main__":

    sources = get_tiles()

    print("Processing sources:")
    print(sources)

    pipe = sources | warp | get_min_max

    min = 0
    max = 0

    for minmax in pipe.results():
        if minmax[0] < min:
            min = minmax[0]
        if minmax[1] > max:
            max = minmax[1]

    print(f"min: {min}, max: {max}")
    print("Done")
