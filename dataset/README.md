# README

## Statistics of the dataset NHANES

[National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/index.htm)

[Official Tutorials](https://wwwn.cdc.gov/nchs/nhanes/tutorials/Datasets.aspx)

## Latest Change
- Added two files `imputed_with_2cluster_raw.csv` and `imputed_with_2cluster.csv`
  - `imputed_with_2cluster_raw.csv` is self-contained, merging the original `imputed_clust_v1.csv` and `imputed_clust_v2.csv`. `cluster_v1` in `imputed_with_2cluster_raw.csv` contains the original contents of `final3` in `imputed_clust_v1.csv`. `cluster_v2` in `imputed_with_2cluster_raw.csv` contains the original contents of `final3` in `imputed_clust_v2.csv`.
  - The class labels of `cluster_v1` and `cluster_v2` were changed from {1,2,3} to {0,1,2}.`imputed_with_2cluster_raw.csv`.
  - `imputed_with_2cluster.csv` is the file actually used for coding. Compared with the raw file, it further added `train_test_split` and mapped the categorical values to numbers.
- The other `.csv` files are all deprecated since this version, including
  - `imputed_clust_v1.csv`
  - `imputed_clust_v2.csv`
  - `imputed.csv`
  - `nhanes_finalm1.csv`

## The Original README content

- `imputed.csv` - the imputed data
  - the variable `_mi` (= 0,...10) indicates original data `(_mi=0)` and imputed data `(_mi=1,..,10)`
- `imputed_clust_v1.csv` - same as (2) but variable `final3` indicates the 'old' cluster assignments
- `imputed_clust_v2.csv` - same as (2) but varaible `final3` indicates the 'new' cluster assignments

### Clustering Details

For each subject, there are **10** assignments, from the 10 imputed datasets - for each dataset I named cluster 1, 2, 3 according to size (large, medium, small), and then assigned the subject to cluster that recieved the plurality of 'votes'. So for instance, if the subject was assigned to the largest cluster 8 times out of 10, I put them in cluster 1. If there was a tie (eg, 5 times to cluster 1, 5 times to cluster 2), I broke the tie using the lowest (largest) cluster.

Reference paper used to create the newest cluster labels <https://link.springer.com/article/10.1007/s00357-022-09422-y>

Code book are the csv files included in the folder

Do not share these content please!
