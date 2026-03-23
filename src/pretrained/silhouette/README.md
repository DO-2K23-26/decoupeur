# Silhouette Extractor — U-Net + ResNet34

Segmentation binaire de silhouettes humaines avec U-Net et un encodeur ResNet34 pré-entraîné sur ImageNet.

## Dataset

AISegment Human Matting — 34 425 images avec masques.

Téléchargement : https://datasetninja.com/aisegmentcom-human-matting


Les annotations sont au format Supervisely JSON. Le masque binaire est extrait depuis le champ `bitmap` encodé en base64+zlib.

## Modèle

U-Net avec encodeur ResNet34 pré-entraîné (`segmentation_models_pytorch`).

L'encodeur compresse l'image 512×512 → 8×8 en extrayant des features multi-niveaux. Le décodeur remonte progressivement jusqu'à 512×512 pour produire le masque pixel par pixel. Les skip connections relient chaque niveau encodeur → décodeur pour préserver les détails fins des contours.

## Entraînement

- optimizer : AdamW, lr=1e-4, weight_decay=1e-4
- scheduler : CosineAnnealingLR
- loss : Dice + BCE
- epochs max : 60, early stopping patience=10
- img_size : 512×512
- batch_size : 16
- AMP : activé (mixed precision)

Augmentations en train : flip horizontal, rotation, elastic deformation, grid distortion, brightness/contrast.

## Résultats

IoU attendu : 80–88% sur le test set.

## GPU

Testé sur NVIDIA L40S. Temps estimé : 2–4 heures.

## Dépendances

```
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python-headless
tqdm
```


## Utilisation

Ouvrir `silhouette.ipynb` et exécuter les cellules dans l'ordre.

Splits train/val/test générés automatiquement dans `result/data/splits/`.
Meilleur checkpoint sauvegardé dans `result/checkpoints/best.pth`.
Métriques loguées dans `result/logs/metrics.csv`.
