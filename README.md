# BSCS23175-ASSIGNMENT02
 Fine-tuning ResNet-50 and U-Net on small image datasets.

ResNet-50 Classification & U-Net Segmentation with Transfer Learning

Assignment 02 — ANN & Deep Learning

This repository contains the complete implementation and results of a transfer-learning assignment in PyTorch. Two ImageNet-pretrained networks are fine-tuned on very small datasets (150–200 images):

Model	Task	Dataset	Main metric	Test result
ResNet-50	Image classification	Oxford Flowers-102 subset (5 classes × 40 images)	Accuracy	100.0 %
U-Net (ResNet-50 encoder)	Binary segmentation	Oxford-IIIT Pet subset (150 image/mask pairs, 37 breeds)	Dice / IoU	0.918 / 0.856
Repository Structure
text
resnet-unet-finetuning-assignment/
│
├── notebook/
│   └── assignment_resnet_unet.ipynb     <- completed notebook, run top to bottom, saved with outputs
│
├── results/
│   ├── flowers_class_distribution.png   <- Oxford Flowers-102 subset: images per class
│   ├── pets_breed_distribution.png      <- Oxford-IIIT Pet subset: pairs per breed
│   ├── resnet_results.png               <- ResNet-50 training / validation curves (loss, accuracy)
│   ├── resnet_confusion_matrix.png      <- ResNet-50 confusion matrix on the test set
│   ├── unet_curves.png                  <- U-Net training curves (loss, Dice, IoU)
│   ├── unet_predictions.png             <- Input | Ground truth | Prediction | Overlay
│   ├── resnet_metrics.csv               <- ResNet-50 metrics table
│   ├── unet_metrics.csv                 <- U-Net metrics table
│   └── summary.csv                      <- comparison of both models
│
├── .gitignore
└── README.md

Datasets and trained weights (*.pth) are not included: GitHub rejects files larger than 100 MB. The checkpoints are stored on Google Drive (MyDrive/ANN_DL_Assignment/checkpoints).

Datasets
	Classification	Segmentation
Dataset	Oxford Flowers-102	Oxford-IIIT Pet
Files used	102flowers.tgz, imagelabels.mat	images.tar.gz, annotations.tar.gz (trimaps)
Subset	5 classes × 40 images = 200 images	150 image/mask pairs, spread over all 37 breeds
Classes	pink primrose, hard-leaved pocket orchid, canterbury bells, sweet pea, english marigold	binary: pet (trimap 1 + 3) vs. background (trimap 2)
Split (70 / 15 / 15)	140 / 30 / 30 — stratified by class	105 / 22 / 23 — stratified by species

The flower classes are read from the official label file imagelabels.mat; the pet breed of every image is detected from its file name (<Breed>_<number>, cat breeds start with a capital letter).

Method
Part A — ResNet-50 Fine-Tuning (Classification)
Preprocessing: images resized to 224 × 224 and normalised with the ImageNet mean/std.
Augmentation (training only): random resized crop, horizontal flip, rotation (±15°) and colour jitter. Validation and test images are never augmented.
Model: ImageNet-pretrained ResNet-50; the 1000-class fc layer is replaced by Dropout + Linear (2048 → 5).
Phase 1 — feature extraction: the whole backbone is frozen, only the new head is trained (AdamW, lr 1e-3, weight decay 1e-4).
Phase 2 — fine-tuning: layer4 is unfrozen and trained together with the head at a 10× smaller lr (1e-4) with cosine annealing.
Training details: cross-entropy with label smoothing (0.1), batch size 16, mixed precision (AMP), frozen BatchNorm layers kept in eval mode, early stopping, and the best validation checkpoint is restored after each phase.
Part B — U-Net Fine-Tuning (Segmentation)
Model: segmentation_models_pytorch U-Net with an ImageNet-pretrained ResNet-50 encoder and 1 output logit per pixel.
Joint spatial transforms: a custom dataset class applies the same resize (256 × 256), horizontal flip and rotation to the image and its mask. Masks use nearest-neighbour interpolation so they stay strictly binary; brightness/contrast changes are applied to the image only.
Loss: BCE + soft Dice loss — per-pixel accuracy plus robustness to foreground/background imbalance.
Metrics: Dice, IoU and pixel accuracy (threshold 0.5).
Phase 1 — decoder training: encoder frozen, decoder + head trained at lr 1e-3.
Phase 2 — discriminative fine-tuning: encoder layer3 and layer4 unfrozen at lr 1e-4, decoder + head at 3e-4, cosine annealing.
Training details: batch size 8, mixed precision, early stopping, best validation Dice checkpoint restored.
Results
ResNet-50 (Classification)
Metric	Result
Training Accuracy	1.0000
Validation Accuracy	0.9667
Test Accuracy	1.0000
Precision (macro)	1.0000
Recall (macro)	1.0000
F1-score (macro)	1.0000
U-Net (Segmentation)
Metric	Result
Training Loss (BCE + Dice)	0.1538
Validation Loss	0.2243
Test Dice	0.9183
Test IoU	0.8561
Test Pixel Accuracy	0.9463
Comparative Summary
Model	Dataset	Task	Main Metric	Result
ResNet-50	Oxford Flowers-102 (200 images, 5 classes)	Classification	Accuracy	1.0000
U-Net	Oxford-IIIT Pet (150 image/mask pairs, 37 breeds)	Segmentation	Dice / IoU	0.9183 / 0.8561
Findings and Discussion Summary
Transfer learning works on tiny datasets. With only 140 training images, ResNet-50 classified all 30 test images correctly (validation 96.7 %, i.e. one mistake). The U-Net reached a test Dice of 0.918 from 105 training pairs.
Small test sets → high variance. With 30 flower test images, one image equals 3.3 % accuracy, so the 100 % test score should be read as "very high" rather than "perfect". K-fold cross-validation would give a more reliable estimate.
Augmentation only on training data. Keeping eval_tf deterministic makes validation and test scores consistent and reproducible.
Nearest-neighbour for masks. Bilinear/bicubic interpolation averages neighbouring pixels and creates invalid in-between labels along the boundary; nearest-neighbour keeps the mask strictly binary.
Two-phase fine-tuning prevents catastrophic forgetting. The randomly initialised head produces large, noisy gradients; freezing the backbone first protects the pretrained features. Unfreezing only the deepest stages (layer4, or layer3 + layer4 for the U-Net) with a small learning rate keeps generic low-level filters (edges, textures) and adapts only the task-specific layers.
Frozen BatchNorm in eval mode. Prevents tiny mini-batches from overwriting the pretrained running statistics.
BCE + Dice loss. BCE gives stable per-pixel gradients; Dice optimises the overlap directly and is robust to small foreground objects.
Overfitting. The gap between training and validation loss of the U-Net (0.154 vs. 0.224) is small; augmentation, weight decay, early stopping and restoring the best epoch kept overfitting under control.
Where the U-Net fails. Errors concentrate on soft fur boundaries and thin structures (ears, legs, tails), which lose detail after the encoder's 32× down-sampling, and on backgrounds with a colour/texture similar to the fur.
