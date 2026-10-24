# Rice Leaf Disease Classification (Classical Machine Learning)

> 🏁 Warm-up round of **FPTU Can Tho Olympic AI 2026** (Kaggle competition) – solved with **classical ML only**, no CNN / deep learning.

Classifies rice leaf images into **8 classes** (Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, Narrow Brown Leaf Spot, Rice Hispa, Sheath Blight) using hand-crafted image features and an ensemble of tree-based models and SVM.

## Approach

**1. Feature extraction (≈10,400 features per image)**
- CLAHE contrast enhancement
- Multi-scale **HOG**
- Multi-radius **LBP** and **GLCM** texture features
- Color moments and **Hu moments**
- **Leaf/lesion segmentation**, then lesion-specific color (Lab) and texture features

**2. Preprocessing and feature selection (inside a `Pipeline`)**
`StandardScaler` → `VarianceThreshold` → `SelectKBest` (k=500) → `SelectFromModel`

**3. Models**
- SVM (RBF kernel), tuned with `RandomizedSearchCV`
- Random Forest, tuned with `RandomizedSearchCV`
- Extra Trees
- Soft **Voting** and **Stacking** ensembles of the three base models

**4. Evaluation**
Stratified 5-fold cross-validation to pick the final model, retrained on the full training set for the submission.

## Results

| Model (single 80/20 validation split) | Accuracy | Macro F1 |
|---|---|---|
| Stacking | 91.1% | 0.903 |
| SVM | 90.6% | 0.891 |
| Soft Voting | 90.4% | 0.890 |
| Extra Trees | 89.4% | 0.883 |
| Random Forest | 89.2% | 0.874 |

Leaderboard score (submitted after the deadline): **0.75146** (private) / **0.72749** (public). The gap between validation and leaderboard score suggests the hidden test set is harder or differently distributed than the training data.

## Dataset and running

The competition dataset (2,026 training images) is **not included** in this repository. The notebook was run on Kaggle and reads data from `/kaggle/input/...`. To run it elsewhere, change the `DATASET` path in the second cell to your local copy.

```bash
pip install -r requirements.txt
jupyter notebook rice_leaf_classification.ipynb
```

## Tech stack

Python · OpenCV · scikit-image · scikit-learn · NumPy · Pandas · Jupyter
