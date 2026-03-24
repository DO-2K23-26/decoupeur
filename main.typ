#import "@preview/typslides:1.3.2": *

// Project configuration
#show: typslides.with(
  ratio: "16-9",
  theme: "reddy",
  font: "Lora",
  font-size: 20pt,
  link-style: "color",
  show-progress: true,
)

// The front slide
#front-slide(
  title: "Automatic background removal using convolutional networks",
  subtitle: [Using _pytorch_],
  authors: "D. Tetu, P. Contat, D. Grasset, T. Radulescu",
  info: [#link("https://github.com/DO-2K23-26/decoupeur")],
)

// ============================================================
// INTRODUCTION
// ============================================================

#slide(title: "Introduction - What's background removal ?", outlined: true)[
  #figure(
  image("images/6.jpg", width: 50%),
  caption: [
    Foreground extraction of a cat
  ],
)
  *Goal* : Isolate an element from an image as accurately as possible
]

#slide(title: "Why using machine learning for that ?", outlined: true)[
  - A lot of different image parameters (lighting, contrast, blurriness...).
  - A lot of different poses.
  - An infinity of different backgrounds.
  - Almost impossible to code an algorithm for every cases.
]


// Custom outline
#table-of-contents()

// ============================================================
// State of the art
// ============================================================


#slide(title: "State of the art", outlined: true)[
  = Real-Time High-Resolution Background Matting [2020]
  A whitepaper by students at University of Washington
  
  #figure(
  image("images/architecture_state_of_the_art.png", width: 100%),
  caption: [
    Architecture of their model
  ]
)
]


// ============================================================
// PRETRAINED SILHOUETTE SECTION
// ============================================================

#title-slide[
  Pretrained Silhouette Extractor
]

// Motivation
#slide(title: "Why Transfer Learning for Silhouette Segmentation?", outlined: true)[
  *Problem statement:*
  - 34,425 images in dataset
  - U-Net + ResNet34 = 13.4M parameters
  - Ratio: 2.5 images per parameter #sym.arrow severe underfitting risk without transfer learning

  *Transfer Learning Benefits:*
  - ImageNet features already learned (edges, textures, shapes)
  - Faster convergence: 30--50 epochs vs 100--150 from scratch
  - Better generalization: lower overfitting risk
  - 4x GPU cost reduction

  #framed(title: "Key insight")[
    Features learned on 1.2M ImageNet images transfer well to silhouette extraction.
  ]
]

// Architecture Overview
#slide(title: "U-Net Architecture with ResNet34 Encoder", outlined: true)[
  *Encoder Path (Compression):*
  - Input: (3, 512, 512)
  - Pretrained ResNet34 backbone
  - Progressively downsamples: 512 #sym.arrow 256 #sym.arrow 128 #sym.arrow 64 #sym.arrow 8 (spatial dims)
  - Extracts multi-level semantic features

  *Bottleneck:*
  - Features at 8x8 resolution
  - Captures global context without spatial precision

  *Decoder Path (Reconstruction):*
  - Transposed convolutions: 8 #sym.arrow 16 #sym.arrow 32 #sym.arrow 64 #sym.arrow 128 #sym.arrow 256 #sym.arrow 512
  - Gradually upsamples to original resolution
  - Output: (1, 512, 512) binary mask

  #framed(title: "Skip Connections")[
    Connect encoder layers directly to corresponding decoder layers.
    Preserves fine-grained boundary details during upsampling.
  ]
]

// Training Configuration
#slide(title: "Training Configuration", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Hyperparameters:*
    - Image size: 512x512
    - Batch size: 16
    - Learning rate: 1e-4
    - Weight decay: 1e-4
    - Optimizer: AdamW
    - Scheduler: CosineAnnealingLR
    - Max epochs: 60
    - Early stopping: patience=10
  ][
    *Regularization:*
    - Discriminative LR per layer
    - Mixed precision (AMP)
    - Data augmentation:
      - Horizontal flip (50%)
      - Rotation (+-15 deg)
      - Elastic deformations
      - Grid distortion
      - Brightness/contrast
      - Dropout (spatial)
  ]

  #framed(title: "Loss Function")[
    Loss = 0.5 x DiceLoss + 0.5 x BCELoss --- Dice handles class imbalance (99% background pixels), BCE stabilizes convergence.
  ]
]

// Fine-tuning Strategy
#slide(title: "Discriminative Layer Learning Rates", outlined: true)[
  *Principle:* Lower learning rates for early layers (preserve ImageNet features), higher for later layers and decoder.

  #table(
    columns: (2fr, 1fr, 1.5fr),
    [*Layer Group*], [*LR*], [*Rationale*],
    [Conv1 (edges/gradients)], [1e-5], [Freeze near-completely],
    [Layer1 (textures)], [1e-4], [Small updates],
    [Layer2 (shapes low)], [1e-3], [Larger updates],
    [Layer3 (shapes high)], [1e-3], [Learn silhouette-specific features],
    [Decoder], [1e-3], [Train from scratch on silhouettes],
  )

  #grayed[*Benefit:* Balances preserving ImageNet knowledge with adapting to silhouette task.]
]

// Evaluation Metrics
#slide(title: "Evaluation Metrics", outlined: true)[
  *Intersection over Union (IoU):*
  - IoU = |Pred #sym.inter True| / |Pred #sym.union True|
  - Range: [0, 1], higher is better
  - Insensitive to class imbalance
  - Standard metric for segmentation

  *Dice Score (F1-score):*
  - Dice = 2 x |Pred #sym.inter True| / (|Pred| + |True|)
  - Comparable to IoU, also used as loss function

  *Pixel Accuracy:*
  - % of correctly classified pixels
  - #stress[Warning:] Misleading alone --- can achieve 99% by predicting all background

  #framed(title: "Primary metric: IoU on validation set")[
    Used for early stopping and model selection.
  ]
]

// ============================================================
// RESULTS SECTION
// ============================================================

#slide(title: "Qualitative Results: Pretrained Model", outlined: true)[
  *Examples of successful segmentations:*

  #framed(back-color: rgb("f0f0f0"))[
    #align(center)[
      _Insert here: 4 examples side by side (input | ground truth | prediction | overlay)_
      #v(3cm)
    ]
  ]

  *Observations:*
  - Sharp, precise contours
  - Handles varying lighting conditions
  - Robust to complex poses and occlusions
]

#slide(title: "Quantitative Results: Pretrained Model", outlined: true)[
  *Metrics on validation set (split: 70% train / 15% val / 15% test):*

  #table(
    columns: (2fr, 1fr, 1fr, 1fr),
    [*Metric*], [*Mean*], [*Std Dev*], [*Range*],
    [IoU], [(insert)], [(insert)], [(insert)],
    [Dice], [(insert)], [(insert)], [(insert)],
    [Pixel Accuracy], [(insert)], [(insert)], [(insert)],
  )

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Training dynamics:*
    - Epochs to convergence: (insert)
    - Best epoch: (insert)
    - Final validation loss: (insert)
  ][
    *Computational cost (school GPU nodes):*
    - Training time: (insert) hours
    - Inference time / image: (insert) ms
    - Model size: (insert) MB
  ]
]

// ============================================================
// COMPARISON SECTION
// ============================================================

#slide(title: "Transfer Learning vs From Scratch", outlined: true)[
  *Hypothesis:* Pretrained ResNet34 should outperform training from scratch on limited data.

  #table(
    columns: (2fr, 1fr, 1fr),
    [*Aspect*], [*From Scratch*], [*Pretrained (ResNet34)*],
    [Epochs to convergence], [100--150], [(insert)],
    [Final IoU (validation)], [72--78% (est.)], [(insert)],
    [Training time (GPU hours)], [12--18], [(insert)],
    [Overfitting risk], [High], [Low],
    [GPU node usage], [High], [(insert)],
  )

  #framed(title: "Efficiency gain")[
    Pretrained model achieves (insert)% higher IoU with (insert)x faster training on school GPU nodes.
  ]
]

#slide(title: "Visual Comparison: From Scratch vs Pretrained", outlined: true)[
  *From-scratch model results:*

  #framed(back-color: rgb("ffe0e0"))[
    #align(center)[
      _Insert here: input | prediction | ground truth_
      #v(2cm)
    ]
  ]

  *Pretrained model results:*

  #framed(back-color: rgb("e0ffe0"))[
    #align(center)[
      _Insert here: input | prediction | ground truth_
      #v(2cm)
    ]
  ]
]

// Error Analysis
#slide(title: "Error Analysis and Failure Cases", outlined: true)[
  *Common failure modes:*
  - Occlusion: overlapping people or objects
  - Extreme poses: contorted silhouettes beyond training distribution
  - Low contrast: silhouettes blending into background

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Example 1: Occlusion*
    #framed(back-color: rgb("f0f0f0"))[
      #align(center)[
        _Insert failed example_
        #v(2cm)
      ]
      IoU: (insert)
    ]
  ][
    *Example 2: Low contrast*
    #framed(back-color: rgb("f0f0f0"))[
      #align(center)[
        _Insert failed example_
        #v(2cm)
      ]
      IoU: (insert)
    ]
  ]
]

// Conclusion
#slide(title: "Conclusions: Pretrained Silhouette Extractor", outlined: true)[
  *Findings:*
  - Transfer learning from ImageNet enables robust silhouette segmentation on 34K images
  - Achieves (insert)% IoU with efficient training on school GPU nodes
  - Significantly outperforms from-scratch baseline in speed and accuracy

  *Best practices applied:*
  - Discriminative layer-wise learning rates
  - Dice + BCE combined loss for class imbalance
  - Aggressive data augmentation
  - Early stopping and model checkpointing

  *Future improvements:*
  - Multi-scale inference (pyramid approach)
  - Ensemble of multiple architectures
  - Real-time optimization for edge deployment
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
