#import "@preview/typslides:1.3.2": *

#set bibliography(title: none, full: true)

#show: typslides.with(
  ratio: "16-9",
  theme: "reddy",
  font: "Linux Libertine",
  font-size: 20pt,
  link-style: "color",
  show-progress: true,
)

#front-slide(
  title: "Automatic background removal using convolutional networks",
  subtitle: [Using _pytorch_],
  authors: "D. Tetu, P. Contat, D. Grasset, T. Radulescu",
  info: [#link("https://github.com/DO-2K23-26/decoupeur") \
    #link("https://huggingface.co/Courtcircuits/decoupeur")],
)

// ============================================================
// INTRODUCTION
// ============================================================

#slide(title: "Introduction - What's background removal ?", outlined: true)[
  #figure(
    image("images/6.jpg", width: 50%),
    caption: [Foreground extraction of a cat],
  )
  *Goal* : Isolate an element from an image as accurately as possible
]

#slide(title: "Why using machine learning for that ?", outlined: true)[
  - A lot of different image parameters (lighting, contrast, blurriness...).
  - A lot of different poses.
  - An infinity of different backgrounds.
  - Almost impossible to code an algorithm for every cases.
]

#table-of-contents()

// ============================================================
// State of the art
// ============================================================

#slide(title: "State of the art", outlined: true)[
  = Real-Time High-Resolution Background Matting [2020]@BMSengupta20
  Soumyadip Sengupta and Vivek Jayaram and Brian Curless and Steve Seitz and Ira Kemelmacher-Shlizerman

  #figure(
    image("images/architecture_state_of_the_art.png", width: 100%),
    caption: [Architecture of their model],
  )
]

// ============================================================
#title-slide[
  Pretrained Background Remover (Transfer Learning)
]

// --- WHY TRANSFER LEARNING ---
#slide(title: "Why Transfer Learning?", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Our dataset:* 34,425 labeled images of people

    *The challenge:* U-Net with a ResNet34 backbone has *24.4M parameters*. These parameters need to learn visual features from data: edges, textures, object shapes, and human silhouettes.
  ][
    *Transfer Learning solution:* reuse ResNet34 pretrained on ImageNet (1.2M images). It already recognizes fundamental visual patterns. We only finetune it for human vs. background distinction.

    *Benefits:*
    + Converges in 16 epochs (~30 minutes on school GPUs)
    + Better results with limited labeled data
    + Lower compute cost
    + Improved generalization
  ]
]

// --- U-NET ARCHITECTURE ---
#slide(title: "What is U-Net ?", outlined: true)[
  #figure(
    image("images/unet.png", width: 70%),
    caption: [Architecture of U-Net],
  )
]

#slide(title: "Why U-Net? Architecture and Layers", outlined: true)[
  *Goal:* for each pixel of a 512x512 image, decide: human (1) or background (0)?

  A standard CNN classifies a whole image into one category. U-Net is specifically designed for pixel-level segmentation: it outputs a mask of the same size as the input. This makes it the standard architecture for biomedical and human segmentation tasks.

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    At each stage: convolutions + ReLU activations + max-pooling halve the spatial size and double the number of feature maps.
  ][
    Each decoder block uses transposed convolution to double spatial size, then concatenates with the matching encoder stage output.
  ]
  #align(center)[
    #framed(title: "Why are they called Skip Connections?")[
      At each level, the encoder output *skips* the bottleneck and connects directly to the corresponding decoder level. This reintroduces the fine spatial details (edges, contours) lost during compression.
    ]
  ]
]

// --- TRAINING CONFIG ---
#slide(title: "Training Configuration", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Setup:*
    + Image size: 512x512
    + Batch size: 16
    + Epochs trained: *16*
    + Mixed precision (AMP): uses 16-bit floats instead of 32-bit where safe, halving memory usage and significantly speeding up GPU operations, with no accuracy loss
  ][
    *Optimizer: AdamW*
    + Adapts learning rate per parameter automatically
    + Weight decay built-in: penalizes large weights to reduce overfitting
    + More stable convergence than plain SGD on this type of task
  ]
]

// --- LOSS FUNCTION ---
#slide(title: "Loss Function: Dice + BCE", outlined: true)[
  A typical 512x512 photo of a person contains roughly 260,000 pixels of background and only about 2,000 pixels of human (less than 1%). This is called *class imbalance*: one category vastly outnumbers the other.

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Binary Cross-Entropy (BCE):*
    + Scores each pixel independently
    + A model predicting "all background" scores 99% accuracy
    + Completely fooled by class imbalance
    + But: stabilizes learning at the start
  ][
    *Dice Loss:*
    + Measures the overlap between predicted mask and true mask
    + A model predicting "all background" scores 0 (no overlap)
    + Not fooled by class imbalance
    + But: slower to get started alone
  ]

  #framed(title: "Combined: Loss = 0.5 x BCE + 0.5 x Dice")[
    BCE gets training started quickly. Dice ensures the model actually learns to find the human pixels. Together they work better than either alone.
  ]
]

// --- WHY IoU ---
#slide(title: "Why IoU as Evaluation Metric?", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *IoU (Intersection over Union)* measures how much the predicted mask overlaps with the true mask.

    + IoU = 1.0: prediction is perfect
    + IoU = 0.0: prediction and truth do not overlap at all
    + A model predicting all background scores IoU = 0, not 99%
    + Not fooled by class imbalance

  ][
    *Why not Dice then?* Dice and IoU measure very similar things (Dice = 2 x IoU / (1 + IoU)), but IoU is the standard reference in every published segmentation paper. We use Dice as the *loss* (because its gradient is better for optimization) and IoU as the *metric* (because it is the community standard for comparison).
  ]
]

// ============================================================
#title-slide[
  From-Scratch U-Net
]

// --- WHY FROM SCRATCH? ---
#slide(title: "From-Scratch U-Net: Motivation", outlined: true)[
  In parallel, we trained a *custom U-Net implemented entirely from scratch* in PyTorch, without any pretrained backbone.

  *Goal:* measure the actual contribution of transfer learning by comparing both approaches under identical conditions.

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Dataset:*
    #rect()[Warning: not the same dataset as pretrained model!]
    + 40,000 paired image/mask samples
    + Same 512x512 resolution
    + Same augmentation pipeline

    *Why is this harder without pretraining?*
    + The model must learn all visual features from zero
    + Edges, textures, shapes, human contours: nothing is given
    + Requires more data and more epochs to reach the same quality
  ][
    #framed(title: "Key difference vs pretrained")[
      The pretrained model starts with 21.8M parameters already tuned on 1.2M images. The from-scratch model starts with *31M parameters* all randomly initialized, and must discover structure entirely from the training data.
    ]
  ]
]

// --- ARCHITECTURE FROM SCRATCH ---
#slide(title: "Custom U-Net Architecture", outlined: true)[
  #framed(title: "Parameters: 31,043,521")[
    More parameters than the pretrained model (24.4M), but all initialized randomly (which makes training harder and slower).
  ]
]

// --- TRAINING CONFIG FROM SCRATCH ---
#slide(title: "Training Configuration (From Scratch)", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Setup:*
    + Image size: 512x512
    + Batch size: 16
    + Max epochs: 50
  ][
    *Optimizer: Adam*
    + Adapts the learning rate per parameter automatically
    + Similar to AdamW but without built-in weight decay
  ]
]

#title-slide[
  Comparison of Results
]

// --- QUANTITATIVE RESULTS ---
#slide(title: "Quantitative Results", outlined: true)[
  *Metrics on validation set (split: 70% train / 15% val / 15% test):*

  #table(
    columns: (2fr, 1fr, 1fr),
    [*Metric*], [*Pretrained*], [*From Scratch*],
    [IoU], [0.9856], [0.9826],
    [Dice], [0.9927], [0.9912],
    [Epochs], [16], [40],
    [Training time], [~30 min], [~23h],
  )

  #framed(title: "Key result")[
    Pretrained model reaches higher IoU in fewer epochs, with lower GPU usage on school nodes.
  ]
]

// --- VISUAL COMPARISON (1/2) ---
#slide(title: "Visual Comparison: Pretrained vs From Scratch (1/2)", outlined: true)[
  #figure(
    image("images/predictions1.png", width: 70%),
    caption: [Predictions comparison : pretrained (top) vs from-scratch (bottom)],
  )
]

// --- VISUAL COMPARISON (2/2) ---
#slide(title: "Visual Comparison: Pretrained vs From Scratch (2/2)", outlined: true)[
  #figure(
    image("images/predictions2.png", width: 70%),
    caption: [Predictions comparison : pretrained (top) vs from-scratch (bottom)],
  )
]


#slide(title: "Models & Dataset Comparison")[
  #align(center)[
    #framed(back-color: rgb("f0f0f0"))[
      #figure(
        image("images/dice.png", width: 70%),
        caption: [Performances per dataset illustrated],
      )
    ]
  ]
]

// --- DEMO ---
#title-slide[
  Demo Time!
]

#title-slide[
  Conclusion
]


// --- CONCLUSION ---
#slide(title: "Conclusion", outlined: true)[
  *What we built:* a background remover trained on 34K–40K images using two approaches: U-Net + pretrained ResNet34, and a custom U-Net from scratch.

  *Why transfer learning works:*
  + ResNet34 compensates for limited data and trains in ~30 min vs ~23h
  + U-Net reconstructs a full-resolution mask via skip connections
  + Dice + BCE loss handles the extreme imbalance between human and background pixels
  + IoU is a meaningful metric: not fooled by the dominant background class

  *Results:*
  + Pretrained: 0.9856 IoU in only 16 epochs (~30 min)
  + From scratch: 0.9826 IoU after 40 epochs (~23h)
  + Transfer learning: *3x faster convergence, same data, better score*
]

// Bibliography
#slide(title: "References", outlined: true)[
  #bibliography("bibliography.bib")
]
