# Compute Histogram

Computes histogram for a set of tiles. We currently assume that tiles are of type FLOAT.

Output is a file `histogram.csv` with two columns: `pixel value` and `pixel count`

## Bins
Users have two option to bin data: Linear or Log.

### Linear
Linear will multiply float values by 100 and convert to integer. Number of bins will be ((int(max *100) + 10) - (int(min * 100) -10))
This method works best for equally distributed values

### Log
Log fist compute log for values and then compute number of bins.
Number of bins is calculated as (log(max + offset) * 1000) - (log(min + offset) * 1000). Where offset is 0 unless min is negative. If negative offset is abs(min) + 1

# Installation
`./scripts/setup.sh`

#Usage
CLI
```
Usage: compute_histogram [OPTIONS] TILES

Options:
  -m, --method TEXT      Method for creating bins
  -w, --workers INTEGER  Number of parallel workers
  --minmax_only          Only compute minmax, not histogram
  --help                 Show this message and exit.
```

