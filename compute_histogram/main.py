from parallelpipe import stage
import rasterio
import numpy as np
import os
import csv


HISTO_RANGE = (0, 1501)
BINS = len(range(HISTO_RANGE[0], HISTO_RANGE[1]))
MAX_BLOCK_SIZE = 4000
WORKERS = 25
QSIZE = 5
PATHS = [
    "/vsis3/gfw-files/2018_update/biodiversity_significance/{tile_id}.tif",
    "/vsis3/gfw-files/2018_update/plantations/{tile_id}.tif",
]
TILE_CSV = "csv/tiles.csv"


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

        with rasterio.open(source[0]) as src1:
            w = (np.log(src1.read(1)) * 100).astype(np.int16)
        w_m = _apply_mask(_get_mask(w, 0), w)
        histo = _compute_histogram(w_m, BINS, HISTO_RANGE)
        w_m = None
        if source[1] is not None:
            with rasterio.open(source[1]) as src2:
                mask_w = np.invert(src2.read(1).astype(np.bool_))
            w = _apply_mask(_get_mask(w, 0, mask_w), w)
            mask_w = None
            histo_m = _compute_histogram(w, BINS, HISTO_RANGE)
            w = None
            yield (histo, histo_m)
        else:
            w = None
            yield (histo, histo)


def get_sources(tiles):

    sources = list()

    for tile in tiles:

        row, col, min_x, min_y, max_x, max_y = tile
        tile_id = "{}_{}".format(_get_top(int(max_y)), _get_left(int(min_x)))
        path1 = PATHS[0].format(tile_id=tile_id)
        path2 = PATHS[1].format(tile_id=tile_id)

        try:
            with rasterio.open(path1) as src1:
                width = src1.width
                height = src1.height
                left, bottom, right, top = src1.bounds

                print("Found " + path1)

        except Exception:
            print("Could not find " + path1)
        else:
            try:
                with rasterio.open(path2) as src2:
                    assert width == src2.width, "Input rasters must have same dimensions. Abort."
                    assert (height == src2.height), "Input rasters must have same dimensions Abort."

                    s_left, s_bottom, s_right, s_top = src2.bounds

                    assert round(left, 4) == round(s_left, 4), "Input rasters must have same bounds. Abort."
                    assert round(bottom, 4) == round(s_bottom, 4), "Input rasters must have same bounds. Abort."
                    assert round(right, 4) == round(s_right, 4), "Input rasters must have same bounds. Abort."
                    assert round(top, 4) == round(s_top, 4), "Input rasters must have same bounds. Abort."

            except Exception:
                print("Could not find " + path2)
                sources.append([path1, None])
            else:
                print("Found " + path2)
                sources.append([path1, path2])

    return sources


def get_tiles():

    tiles = list()

    dir = os.path.dirname(__file__)
    with open(os.path.join(dir, TILE_CSV)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            tiles.append(row)

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


if __name__ == "__main__":

    tiles = get_tiles()
    sources = get_sources(tiles)

    print("Processing sources:")
    print(sources)

    pipe = sources | process_sources

    first = True
    result = None
    result_m = None
    for histo in pipe.results():
        if first:
            result = histo[0]
            result_m = histo[1]
            first = False
        else:
            result = add_histogram(result, histo[0])
            result_m = add_histogram(result_m, histo[1])

    bins = np.array([x/100 for x in range(HISTO_RANGE[0], HISTO_RANGE[1])])
    histogram = np.vstack((bins, result)).T
    histogram_m = np.vstack((bins, result_m)).T

    print("Histogram: ")
    print(histogram)

    np.savetxt("histogram.csv", histogram, fmt='%1.2f, %d')
    np.savetxt("histogram_masked.csv", histogram_m, fmt='%1.2f, %d')
