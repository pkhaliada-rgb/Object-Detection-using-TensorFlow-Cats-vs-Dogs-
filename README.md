# Object-Detection-using-TensorFlow-Cats-vs-Dogs-

<video src="Object Detection using TensorFlow (Cats vs Dogs).webm" width="100%" controls autoplay loop muted>
  Your browser does not support the video tag.
</video>

A single Jupyter notebook that builds a complete cat/dog detection pipeline from scratch, then wraps it in a live web dashboard — all in one file.

- **Classification** — a convolutional neural network (CNN) trained from scratch on real cat/dog photos
- **Localization** — a pretrained SSD MobileNet V2 model (TensorFlow Hub, trained on COCO), used only to find *where* the animal is
- **Dashboard** — a Flask app, launched and displayed inline at the end of the same notebook, where you can upload any photo and see it detected live

No separate scripts, no multi-file project structure — everything runs from `Object_Detection_TensorFlow.ipynb`.

---

## Why two models?

Public cat/dog photo datasets (like the one used here) provide a class label — *cat* or *dog* — but not a bounding box. Nobody has hand-drawn a box around the animal in every training photo. Producing that data ourselves isn't practical, so this project splits the problem in two:

| Task | How it's solved |
|---|---|
| **What is it?** (classification) | A CNN trained from scratch on real cat/dog images |
| **Where is it?** (localization) | A pretrained detector (SSD MobileNet V2), used only for its bounding box output — its own classification guess is discarded |

The pretrained model's job ends at "there's an animal-shaped thing here." Everything after that — is it a cat or a dog — is decided by the model this project trains itself.

---

## What's inside the notebook

| Section | What it does |
|---|---|
| 1 — Imports and Setup | Fast dependency check, core imports, Colab detection |
| 2 — Visualization Utilities | Reusable functions for drawing bounding boxes on any image |
| 3 — Data Extraction and Preprocessing | Streams a balanced set of cat/dog photos from Hugging Face Hub |
| 4 — Building the CNN | A from-scratch Conv2D/MaxPooling/Dense architecture |
| 5 — Training and Evaluation | Trains the CNN, plots loss/accuracy, visualizes predictions |
| 6 — Pretrained Locator | Loads SSD MobileNet V2, used only to find the animal's box |
| 7 — Combining Both | Locate → crop → classify with the CNN, with a graceful fallback |
| 8 — The Dashboard | Runs a Flask app in a background thread, displayed inline in the notebook |

---

## Getting started

### Requirements

- Python 3.9–3.11
- Jupyter Notebook, JupyterLab, VS Code (with the Jupyter extension), or Google Colab

### Setup (PowerShell / any terminal)

```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
venv\Scripts\activate        # on macOS/Linux: source venv/bin/activate

pip install --upgrade pip
```

No `requirements.txt` is needed to get started — the notebook's first cell checks for every required package and installs anything missing automatically the first time you run it.

### Running it

1. Open `Object_Detection_TensorFlow.ipynb` in your editor of choice.
2. Run the cells from top to bottom (**Run All** works fine).
3. The first run downloads the pretrained locator model and streams a small set of training images — this can take a few minutes depending on your connection. Every cell after that is fast, and safe to re-run.
4. When you reach the final cell, a live dashboard renders directly inside the notebook. Upload any photo of a cat or dog and see it detected and classified.

**Running in Google Colab:** works the same way — the notebook automatically detects Colab and uses its port-forwarding to display the dashboard correctly.

---

## The dashboard

<p align="center">
  <!-- Add a screenshot at docs/dashboard-screenshot.png to show it here -->
  <img src="docs/dashboard-screenshot.png" alt="Dashboard screenshot" width="500">
</p>

The dashboard is a small Flask app defined in the notebook's final section. It:

- Runs in a background thread, so it doesn't block the rest of the notebook
- Accepts an uploaded photo, runs it through the same `detect_and_classify` pipeline built earlier in the notebook
- Draws a bounding box around the detected animal and reports the CNN's label and confidence
- Encodes results as an embedded image (base64 data URI) — nothing is ever written to disk

No new detection logic is introduced in the dashboard section; it's a thin presentation layer over the pipeline already built in Sections 6–7.

---

## Dataset

Training data is streamed from [`microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) on Hugging Face Hub — roughly 2,500 images are pulled (not the full ~23,000), split evenly between cats and dogs.

**Note:** this dataset's rows are ordered by class (all cats, then all dogs). The notebook explicitly collects a balanced number of each class rather than taking a naive slice — a naive `.take(N)` would silently grab almost entirely one class, which trains a model that just learns to always guess that class. A verification step prints the resulting class counts so this is easy to confirm.

If Hugging Face Hub is unreachable on your network, the notebook prints a fallback pointing to an equivalent dataset on Kaggle: [`salader/dogs-vs-cats`](https://www.kaggle.com/datasets/salader/dogs-vs-cats).

---

## Model details

**Classifier (trained in this notebook):**
- 3× `Conv2D` + `MaxPooling2D` blocks (32 → 64 → 128 filters)
- `Flatten` → `Dense(128, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- Binary cross-entropy loss, Adam optimizer
- Trained for 5 epochs on ~2,000 images by default — enough to see the pipeline work end to end, not tuned for maximum accuracy. See **Next steps** below.

**Locator (pretrained, not modified):**
- [SSD MobileNet V2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2) from TensorFlow Hub, trained on COCO
- Used only to find animal-shaped regions (COCO classes: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe) above a confidence threshold

---

## Next steps

- Increase `EPOCHS`, `N_TRAIN`, and `N_VAL` in Section 5 / Section 3 for a more accurate classifier
- Add data augmentation (random flips, rotations, zooms) to improve generalization from a relatively small training set
- Swap the locator for a different TensorFlow Hub detection model and compare speed/accuracy
- Extend the CNN to more than two classes (e.g. specific dog/cat breeds)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | Run `pip install "setuptools<81"` — newer setuptools versions removed `pkg_resources`, which `tensorflow_hub` still depends on |
| Hugging Face download fails / times out | Check your network can reach `huggingface.co`, or use the Kaggle fallback link printed in the error message |
| Dashboard area renders blank | Re-run the final cell once — the local server occasionally needs a moment to finish starting before the embedded view loads |
| Training loss starts very low and never moves | Almost certainly a class-imbalance issue in the data pipeline — re-run Section 3 and check the printed class balance output |

---

## License

Add a license of your choice (e.g. MIT) before publishing this repository publicly.

## Acknowledgments

- [TensorFlow Hub](https://tfhub.dev/) — SSD MobileNet V2 pretrained detector
- [Hugging Face Hub](https://huggingface.co/datasets/microsoft/cats_vs_dogs) — cats vs dogs dataset
- [Flask](https://flask.palletsprojects.com/) — dashboard web framework
