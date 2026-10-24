# Data

Nothing in `raw/`, `clean/` or `model_ready/` is committed (size); only `sample/` is versioned.

| Folder | Content | How to get it |
|---|---|---|
| `raw/` | the three original labeled CSVs (`dataset_01_*`, `dataset_02_*`, `dataset_03_*`) | copy them here (file names are matched by the `dataset_0N_` prefix) |
| `clean/` | `D{1,2,3}_clean.parquet` - validated, de-duplicated, imputed, all remaining raw columns | `make clean-data` |
| `model_ready/` | `D{1,2,3}_model_ready.{parquet,csv.gz}` - 63 features, future targets, chronological split | `make features` |
| `sample/` | 1,500 consecutive rows of router r2 from D3 (test split) for a quick look at the schema | committed |

Column definitions: [`../docs/data_dictionary.md`](../docs/data_dictionary.md).

## Sharing the generated tables
The three `model_ready.csv.gz` files are about 17-20 MB each and the parquet files about 70 MB in total. Do not commit
them to the main history; attach them to a GitHub *Release*, or track `data/model_ready/*.csv.gz` with Git LFS:

```
git lfs install
git lfs track "data/model_ready/*.csv.gz"
```
(and remove `data/model_ready/*` from `.gitignore` if you want them versioned).
