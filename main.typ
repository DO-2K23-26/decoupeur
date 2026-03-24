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
        #figure(
          image("images/dice.png", width: 130%),
          caption: [
            Foreground extraction of a cat
          ],
        )
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
  *Encoder (4 stages, custom ConvBlocks):*
  + Input: (3, 512, 512) — RGB image
  + Stage 1: 3 #sym.arrow.r 64 feature maps, 512x512 #sym.arrow.r 256x256
  + Stage 2: 64 #sym.arrow.r 128 maps, 256x256 #sym.arrow.r 128x128
  + Stage 3: 128 #sym.arrow.r 256 maps, 128x128 #sym.arrow.r 64x64
  + Stage 4: 256 #sym.arrow.r 512 maps, 64x64 #sym.arrow.r 32x32
  + Bottleneck: 512 #sym.arrow.r 1024 maps at 16x16

  Each ConvBlock = Conv2d #sym.arrow.r BatchNorm #sym.arrow.r ReLU #sym.arrow.r Conv2d #sym.arrow.r BatchNorm #sym.arrow.r ReLU

  *Decoder (4 upsampling blocks):*
  + TransposedConv doubles spatial size at each step
  + Skip connection concatenates encoder features at each level
  + Final Conv2d (1x1) #sym.arrow.r Sigmoid #sym.arrow.r binary mask

  #framed(title: "Parameters: 31,043,521")[
    More parameters than the pretrained model (24.4M), but all initialized randomly — which makes training harder and slower.
  ]
]

// --- TRAINING CONFIG FROM SCRATCH ---
#slide(title: "Training Configuration (From Scratch)", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1.5em)[
    *Setup:*
    + Image size: 512x512
    + Batch size: 16
    + Max epochs: 50
    + Early stopping: patience = 5 (stops if *loss* does not improve for 5 consecutive epochs)
    + Checkpoint saved every epoch, training resumed automatically

    *Loss: BCE + Dice (0.5 / 0.5)*
    + Same combined loss as the pretrained model
    + Ensures fair comparison
  ][
    *Optimizer: Adam*
    + Adapts the learning rate per parameter automatically
    + Similar to AdamW but without built-in weight decay

    *Scheduler: ReduceLROnPlateau*
    + Monitors the training loss
    + Halves the learning rate (factor=0.5) if no improvement after 5 epochs
    + Adapts to stagnation rather than following a fixed cosine curve
  ]
]

// --- QUALITATIVE RESULTS FROM SCRATCH ---
#slide(title: "Results: From-Scratch Model", outlined: true)[
  *Examples on validation set:*

  #framed(back-color: rgb("f0f0f0"))[
    #align(center)[
      _[Placeholder: 3 columns: Input image | Ground Truth mask | Predicted mask]_
      #v(3.5cm)
    ]
  ]

  + Boundaries tend to be less precise than the pretrained model
  + More training time required to reach comparable quality
  + Demonstrates the cost of learning everything from zero
]

// --- QUANTITATIVE RESULTS FROM SCRATCH ---
#slide(title: "Quantitative Results (From Scratch)", outlined: true)[
  *Metrics on validation set:*

  #table(
    columns: (2fr, 1fr, 1fr),
    [*Metric*], [*Pretrained*], [*From Scratch*],
    [IoU], [0.9856], [0.9826],
    [Dice], [(to fill)], [(to fill)],
    [Pixel Accuracy], [(to fill)], [(to fill)],
    [Epochs trained], [16], [40],
    [Training time], [~30 min], [~23h],
  )

  #framed(title: "Takeaway")[
    The pretrained model reaches competitive IoU in fewer epochs and with less compute, demonstrating the value of transfer learning on limited data.
  ]
]