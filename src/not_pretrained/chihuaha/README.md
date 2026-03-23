# Chihuahua vs Muffin — CNN from scratch

Classifieur binaire entraîné from scratch sur le dataset Kaggle [muffin-vs-chihuahua-image-classification](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification).

## Dataset

~6000 images réparties en deux classes : `chihuahua` et `muffin`. https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification

Structure attendue :

```
data/
├── train/
│   ├── chihuahua/
│   └── muffin/
└── test/
    ├── chihuahua/
    └── muffin/
```

## Modèle

CNN custom, pas de poids pré-entraînés. 4 blocs convolutionnels :

```
Conv(3→32)   + BN + ReLU + MaxPool
Conv(32→64)  + BN + ReLU + MaxPool
Conv(64→128) + BN + ReLU + MaxPool
Conv(128→256)+ BN + ReLU + AdaptiveAvgPool
Flatten → Dropout(0.4) → Linear(256→64) → ReLU → Linear(64→1)
```

~1.2M paramètres. Sortie : 1 logit, loss BCEWithLogitsLoss.

## Entraînement

- optimizer : Adam, lr=1e-3
- scheduler : CosineAnnealingLR
- epochs max : 30, early stopping patience=7
- img_size : 128×128
- batch_size : 32

Augmentations appliquées en train : flip horizontal, rotation ±15°, variations de luminosité/contraste, normalisation mean=0.5 std=0.5.

## Résultats

Accuracy attendue sur le test set : 75–85%.

## Dépendances

```
torch
torchvision
albumentations
opencv-python
scikit-learn
seaborn
tqdm
```

## Utilisation

Ouvrir `chihuahua.ipynb` et exécuter les cellules dans l'ordre.

Pour prédire sur une image :

```python
predict('chemin/vers/image.jpg')
```

Le meilleur checkpoint est sauvegardé dans `checkpoints/best.pth`.
