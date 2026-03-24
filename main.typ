#import "@preview/typslides:1.3.2": *

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
#title-slide[
  Pretrained Background Remover
]

// --- WHY TRANSFER LEARNING ---
#slide(title: "Why Transfer Learning?", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Our dataset:* 34,425 labeled images of people

    *The problem:* U-Net + ResNet34 has *24.4M parameters*. Training from scratch means the model must learn everything from zero: what an edge is, what a texture looks like, what a human silhouette is.

    With only 34K images, that is not enough data to learn all of this reliably.
  ][
    *Transfer Learning:* we reuse ResNet34 already trained on ImageNet (1.2M images). It already knows how to "see" basic visual features. We just teach it to distinguish humans from background.

    *Benefits:*
    + Converges in 16 epochs (training took only *~30 minutes* on school GPU nodes)
    + Better results on limited data
    + Lower compute cost
  ]
]

// --- U-NET ARCHITECTURE ---
#slide(title: "Why U-Net? Architecture and Layers", outlined: true)[
  *Goal:* for each pixel of a 512x512 image, decide: human (1) or background (0)?

  A standard CNN classifies a whole image into one category. U-Net is specifically designed for pixel-level segmentation: it outputs a mask of the same size as the input. This makes it the standard architecture for biomedical and human segmentation tasks.

  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Encoder (ResNet34, 5 stages):*
    + Stage 0: 512x512 #sym.arrow.r 256x256 (64 feature maps)
    + Stage 1: 256x256 #sym.arrow.r 128x128 (64 maps)
    + Stage 2: 128x128 #sym.arrow.r 64x64 (128 maps)
    + Stage 3: 64x64 #sym.arrow.r 32x32 (256 maps)
    + Stage 4: 32x32 #sym.arrow.r 16x16 (512 maps) — bottleneck
    At each stage: convolutions + ReLU activations + max-pooling halve the spatial size and double the number of feature maps.
  ][
    *Decoder (4 upsampling blocks):*
    + 16x16 #sym.arrow.r 32x32 (256 maps)
    + 32x32 #sym.arrow.r 64x64 (128 maps)
    + 64x64 #sym.arrow.r 128x128 (64 maps)
    + 128x128 #sym.arrow.r 512x512 (32 maps) #sym.arrow.r 1 map (sigmoid)

    Each decoder block uses transposed convolution to double spatial size, then concatenates with the matching encoder stage output.

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
    + Early stopping: patience = 10 (stops if validation *IoU* does not improve for 10 consecutive epochs)
    + Mixed precision (AMP): uses 16-bit floats instead of 32-bit where safe, halving memory usage and significantly speeding up GPU operations, with no accuracy loss
  ][
    *Optimizer: AdamW*
    + Adapts learning rate per parameter automatically
    + Weight decay built-in: penalizes large weights to reduce overfitting
    + More stable convergence than plain SGD on this type of task

    *Scheduler: CosineAnnealingLR*
    + Learning rate decreases smoothly following a cosine curve over the training run
    + Avoids large corrections near convergence, where the model is already close to optimal
    + Prevents oscillating around the minimum at the end of training
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

    *Why not Dice then?* Dice and IoU measure very similar things (Dice = 2 x IoU / (1 + IoU)), but IoU is the standard reference in every published segmentation paper. We use Dice as the *loss* (because its gradient is better for optimization) and IoU as the *metric* (because it is the community standard for comparison).
  ][
    #framed(back-color: rgb("f0f0f0"))[
      #align(center)[
        _[Placeholder: diagram showing two overlapping shapes, one labeled "Predicted", one "Ground Truth". Highlight the intersection in green and the union in blue. Show formula: IoU = green area / blue area]_
        #v(4cm)
      ]
    ]
  ]
]

// --- QUALITATIVE RESULTS ---
#slide(title: "Results: Pretrained Model", outlined: true)[
  *Examples on validation set:*

  #framed(back-color: rgb("f0f0f0"))[
    #align(center)[
      _[Placeholder: 4 columns: Input image | Ground Truth mask | Predicted mask | Overlay]_
      #v(3.5cm)
    ]
  ]

  + Sharp and precise contours
  + Works across varied poses and lighting conditions
  + Trained in only 16 epochs (~30 minutes on school GPU nodes)
]

// --- QUANTITATIVE RESULTS ---
#slide(title: "Quantitative Results", outlined: true)[
  *Metrics on validation set (split: 70% train / 15% val / 15% test):*

  #table(
    columns: (2fr, 1fr, 1fr),
    [*Metric*], [*Pretrained*], [*From Scratch*],
    [IoU], [(to fill)], [(to fill)],
    [Dice], [(to fill)], [(to fill)],
    [Pixel Accuracy], [(to fill)], [(to fill)],
    [Epochs], [16], [(to fill)],
    [Training time], [~30 min], [(to fill)],
  )

  #framed(title: "Key result")[
    Pretrained model reaches higher IoU in fewer epochs, with lower GPU usage on school nodes.
  ]
]

// --- VISUAL COMPARISON ---
#slide(title: "Visual Comparison: Pretrained vs From Scratch", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *From Scratch:*
    #framed(back-color: rgb("ffe0e0"))[
      #align(center)[
        _[Placeholder: Input | Prediction | Ground Truth]_
        #v(2.5cm)
      ]
      Blurry boundaries, missed details
    ]
  ][
    *Pretrained (ResNet34):*
    #framed(back-color: rgb("e0ffe0"))[
      #align(center)[
        _[Placeholder: Input | Prediction | Ground Truth]_
        #v(2.5cm)
      ]
      Sharp contours, precise mask
    ]
  ]
]

// --- FAILURE CASES ---
#slide(title: "Failure Cases", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Occlusion (overlapping people):*
    #framed(back-color: rgb("f0f0f0"))[
      #align(center)[
        _[Placeholder: failed example]_
        #v(2cm)
      ]
      IoU: (to fill)
    ]
  ][
    *Low contrast (blending into background):*
    #framed(back-color: rgb("f0f0f0"))[
      #align(center)[
        _[Placeholder: failed example]_
        #v(2cm)
      ]
      IoU: (to fill)
    ]
  ]

  Possible improvements: add occlusion examples to training data, fine-tune on hard cases.
]

// --- CONCLUSION ---
#slide(title: "Conclusion", outlined: true)[
  *What we built:* a background remover trained on 34K images using U-Net + pretrained ResNet34 (24.4M parameters).

  *Why it works:*
  + Transfer learning compensates for limited data and trains in ~30 min
  + U-Net reconstructs a full-resolution mask via skip connections
  + Dice + BCE loss handles the extreme imbalance between human and background pixels
  + IoU is a meaningful metric: not fooled by the dominant background class

  *Results:*
  + Converges in only 16 epochs
  + Achieves (to fill)% IoU on validation set
  + Outperforms from-scratch baseline in both speed and accuracy
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
