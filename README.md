# Scoria Cone Classification: SVM vs CNN

Classifying volcanic scoria (cinder) cones from digital elevation model (DEM) hillshade patches, comparing a classical machine learning approach (SVM with hand-crafted features) against a deep learning approach (CNN on raw image patches).

## Goal

Given a small patch of terrain raster, classify it as:
- `1` = cone (contains a scoria cone)
- `0` = not cone (ordinary terrain)

Positive examples are extracted around known cone-base centroids on Mount Etna. Negative examples are randomly sampled terrain patches at least 400 m from any known cone.

## Data

- **Source:** Etna cone-base shapefile (`coni_monogenici.shp`) and the Etna NW hillshade raster, from the CinderConesDataPack.
- Hillshade rasters were used instead of raw DEM because they make terrain shapes (like cone rims) visually clearer.
- Final cleaned dataset: 410 patches, balanced between cone and non-cone examples.

## Two approaches

### 1. SVM with hand-crafted features
Each 256x256 patch is converted into a feature vector describing:
- brightness statistics (mean, std, min, max, quartiles)
- center-vs-surrounding-ring contrast
- edge strength
- local roughness
- blob count and blob radius (via Laplacian-of-Gaussian blob detection)

An SVM with an RBF kernel is trained on these features, with hyperparameters (`C`, `gamma`) tuned via grid search.

### 2. CNN on raw image patches
Instead of hand-crafted features, patches are resized to 64x64 and fed directly into a small convolutional neural network:
- 3 convolution + max-pooling blocks (16 → 32 → 64 filters)
- dense layer with dropout for regularization
- sigmoid output for binary classification
- random flips/rotations used as light data augmentation
- early stopping on validation loss to avoid overfitting on the small dataset

## Results

Both models were evaluated on a 70/30 stratified train/test split (same random seed) and validated with 5-fold stratified cross-validation.

| Model | Accuracy | Precision | Recall | F1 | CV Mean F1 | CV Std |
|---|---|---|---|---|---|---|
| SVM (default) | 0.699 | 0.735 | 0.725 | 0.730 | 0.840 | 0.021 |
| Tuned SVM | 0.724 | 0.733 | 0.797 | 0.764 | 0.848 | 0.021 |
| CNN (raw images) | 0.772 | 0.860 | 0.710 | 0.778 | 0.830 | 0.019 |

**Key finding:** the CNN and the tuned SVM perform comparably overall. On the single held-out test split, the CNN edges ahead on F1, driven by notably higher precision (fewer false positives). Under 5-fold cross-validation — a more reliable estimate — the tuned SVM is marginally ahead. This is a typical result for a dataset of this size (~400 examples): CNNs generally need substantially more data to reliably outperform well-designed hand-crafted features.

## Tech stack

- **Geospatial:** `geopandas`, `rasterio`, `affine`
- **Classical ML:** `scikit-learn` (SVM, StandardScaler, GridSearchCV, cross-validation)
- **Deep learning:** `TensorFlow` / `Keras`
- **Image processing:** `scikit-image` (blob detection, rescaling), `scipy.ndimage`
- **Data handling / viz:** `pandas`, `numpy`, `matplotlib`

## How to run

1. Clone this repo and open `scoria_cone_svm_classification_project.ipynb` in Jupyter.
2. Update the `BASE_DIR` path at the top of the notebook to point to your local copy of the CinderConesDataPack.
3. Run all cells top to bottom. The SVM section runs first, followed by the CNN section which reuses the same sampled cone/non-cone locations for a fair comparison.

## Notes / limitations

- The dataset is relatively small (~400 patches), which favors interpretable, low-parameter models like SVM and limits how much a CNN can benefit from its extra capacity.
- All data comes from a single volcanic field (Etna); generalization to other volcanic settings is untested.
- Class balance was enforced by design (equal positive/negative sampling), which should be kept in mind when interpreting accuracy figures.

## Author

Abdul Hanan
