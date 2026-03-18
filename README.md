# Object Detection Model 🎯

A custom object detection system built from scratch using PyTorch — no shortcuts, no pre-built detection libraries. Every part (the neural network, the loss functions, the training loop) is written by hand.

---

## What Does This Project Do?

This project can look at an image (or a live webcam feed) and **detect objects** in it — drawing boxes around things like people, cars, or whatever you train it on.

Think of it like building a mini version of what powers self-driving cars or security cameras — but written entirely from scratch as a learning project.

---

## How It Works (Simple Version)

The model has three main parts:

1. **Backbone** — looks at the image and extracts features (like edges, shapes, textures). Think of it as the "eyes" of the model.
2. **Neck (FPN)** — combines features at different scales so the model can detect both small and large objects.
3. **Head** — takes those features and actually predicts *where* the objects are (bounding boxes) and *what* they are (class labels).

---

## Project Structure

```
object-detection-model/
│
├── model/               ← The neural network
│   ├── backbone.py      # Extracts features from images
│   ├── neck.py          # Combines features at different scales
│   ├── head.py          # Predicts boxes and class labels
│   └── detector.py      # Puts it all together
│
├── loss/                ← How the model learns from mistakes
│   ├── iou.py           # Measures how accurate the predicted boxes are
│   └── detection_loss.py # Combined training loss
│
├── data/                ← Data loading and augmentation
│   ├── augment.py       # Random flips, color changes, mosaic mixing, etc.
│   └── dataset.py       # Loads images and labels for training
│
├── utils/               ← Helper tools
│   ├── nms.py           # Removes duplicate detections
│   ├── metrics.py       # Measures how good the model is (mAP)
│   ├── anchors.py       # Generates anchor boxes from your dataset
│   └── visualize.py     # Draws boxes on images
│
├── train.py             ← Run this to train the model
├── eval.py              ← Run this to measure how good the model is
├── detect.py            ← Run this to use the model on images/webcam
└── configs/default.yaml ← Settings for training
```

---

## Setup

### Step 1 — Create a virtual environment and install dependencies

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2 — Prepare your dataset

Organize your images and labels like this:

```
data/
  images/
    train/    ← your training images (.jpg)
    val/      ← your validation images (.jpg)
  labels/
    train/    ← labels for training images (.txt)
    val/      ← labels for validation images (.txt)
```

Each label file (`.txt`) should have one line per object in the image:

```
<class_id> <center_x> <center_y> <width> <height>
```

All values are **normalized between 0 and 1** relative to the image size. For example:

```
0 0.5 0.4 0.3 0.6
```

### Step 3 — (Optional) Generate anchors for your dataset

Anchors are reference box shapes that help the model learn faster. You can generate them from your own labels:

```bash
python utils/anchors.py --label-dir data/labels/train --n 9 --img-size 640
```

Copy the output into `configs/default.yaml` under the `anchors:` field.

### Step 4 — Edit the config file

Open `configs/default.yaml` and update it with your settings:

```yaml
data_type: custom
img_dir:   data/images/train
label_dir: data/labels/train
class_names: [person, car, bicycle]   # ← your class names here
img_size:   320        # use 320 for quick tests, 640 for better accuracy
epochs:     10         # start small; use 100+ for real training
batch_size: 4          # increase if your GPU has more memory
num_workers: 0         # keep at 0 on Windows
```

---

## Training

```bash
python train.py --config configs/default.yaml
```

To resume training from where you left off:

```bash
python train.py --config configs/default.yaml --resume runs/train/last.pt
```

Checkpoints are saved automatically in `runs/train/`. The best model is saved as `best.pt`.

---

## Evaluating the Model

Once training is done, check how well it performs:

```bash
python eval.py --checkpoint runs/train/best.pt \
               --img-dir data/images/val \
               --label-dir data/labels/val
```

This will print the **mAP (mean Average Precision)** — the standard way to measure object detection accuracy. A higher number is better.

---

## Running Inference (Using the Model)

```bash
# On a single image
python detect.py --checkpoint runs/train/best.pt --source photo.jpg

# On a folder of images
python detect.py --checkpoint runs/train/best.pt --source images/

# Live webcam feed
python detect.py --checkpoint runs/train/best.pt --source 0
```

If nothing gets detected, try lowering the confidence threshold (useful for testing with an untrained model):

```bash
python detect.py --checkpoint runs/train/best.pt --source photo.jpg --conf 0.01
```

---

## Training on COCO (Large-scale dataset)

If you want to train on the official COCO dataset:

```bash
python train.py --config configs/default.yaml \
                --data-type coco \
                --data-root /path/to/coco \
                --split train2017 \
                --epochs 300
```

---

## Design Choices (What and Why)

Here's a plain-English explanation of some key decisions made in this project:

**CIoU Loss for box regression** — instead of just measuring overlap between predicted and actual boxes, CIoU also penalizes wrong aspect ratios and center distance. This makes the model learn better-shaped boxes, especially for thin/tall objects.

**K-means anchors** — instead of using generic anchor sizes, this project computes anchors specifically from your dataset's label sizes. This means the model starts with better guesses and trains faster.

**Decoupled detection head** — separating the "what is it?" (classification) and "where is it?" (box regression) branches into two paths gives the model more flexibility and typically improves accuracy.

**Mosaic + MixUp augmentation** — during training, images are randomly mixed together in creative ways. This forces the model to detect objects in unusual contexts, making it more robust.

**Objectness bias of −2.0** — this is a small trick so the model doesn't go haywire at the very start of training. Without it, the model would initially predict objects everywhere and struggle to learn anything useful.

---

## Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | Run `eval.py` after training to see your results |
| mAP@0.5:0.95 | Run `eval.py` after training |

> For meaningful results, train for at least 100 epochs on a proper dataset.

---

## Tech Stack

| What | How |
|------|-----|
| Deep learning framework | PyTorch (no Ultralytics) |
| Neural network backbone | Custom CSP-DarkNet |
| Multi-scale feature fusion | Custom FPN |
| Detection heads | Custom YOLO-style decoupled heads |
| Loss function | CIoU + Binary Cross Entropy |
| Duplicate removal | Pure PyTorch NMS |
| Accuracy metric | mAP (hand-implemented) |
| Data augmentation | Mosaic, MixUp, HSV jitter, flips — built with OpenCV + NumPy |
| Dataset formats | COCO JSON and custom YOLO TXT |

---

## License

MIT — feel free to use, modify, and learn from this code.
