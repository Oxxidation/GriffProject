# Real-Time Style Transfer

Turn any photo, or your live webcam, into a painting in the style of a famous artist, instantly.

## What this project does

You give it a picture, say a photo of your dog. You pick an art style, for example Van Gogh's *The Starry Night*. A moment later you get the same dog, still clearly your dog, but painted with Van Gogh's swirling brush strokes and deep blues and yellows.

It knows four styles from 4 very famous paintings:

 Starry Night by Vincent van Gogh. This has thick swirls, deep blue and gold
 The Great Wave by Katsushika Hokusai. Japanese print, clean lines, blue and cream
 The Scream by Edvard Munch Uuneasy strokes in orange and dark blue 
 Composition VII by Wassily Kandinsky. Bright, chaotic, abstract color shapes

**You can mix styles.** A slider blends two styles together, like 70% Van Gogh and 30% Hokusai, and you can watch the picture shift smoothly from one to the other. A second slider controls how strong the effect is

**It's fast enough for live video.** Point your webcam at yourself and you appear as a moving painting in real time.

## How it works

**1. It learned by practice** I did not write instructions like "make brush strokes swirly." Instead, the program was shown about 5,000 everyday photos (food, people, streets, animals) and practiced repainting each one, over and over, getting a little better every time. After about 40 minutes of practice on a graphics card, it learned how to do it on its own.

**2. A "judge" graded every attempt.** To know whether a repainting was good, the project uses a second, already-trained image-recognition program (called VGG16.) It answers two questions about each attempt:
- Is it still the same picture? Does the dog still look like a dog, in the same spot?
- Does it look like the painting? Do the colors, textures and brush strokes match the artwork?

The painter program kept adjusting itself to score better on both questions at once. The judge never changes; it only grades.

**3. One painter has four styles.** Instead of training four separate programs, one program learned all four styles. Each style is stored as a small "settings card" the program swaps in.

**4. Why it's fast.** The older way to do this works on one picture at a time and takes many seconds to minutes per image, because it starts from scratch every time. This project does the slow learning once, ahead of time. After that, painting a new picture is a single quick pass through the neural network which makes this fast enough for video. The project includes the older method too, so the two can be compared side by side.

## Technical Version of How it Works

I used a small PyTorch network that restyles any photo or webcam frame in four art styles, with a blend slider between styles, served through a Gradio app.

- **TransformNet**: a feed-forward encoder
- **Perceptual loss**: a frozen ImageNet VGG16 measures the content
- 
## Getting started - Written by Claude
## I chose claude to help with this part because I needed my project to be easily run on other computers

There are two ways to use this project:

- **Use the app** (about 10 minutes): install it and start restyling photos or your webcam with the model I already trained. No training needed.
- **Do it yourself** (about 1 hour): download the practice photos and train your own model from scratch, step by step.

### What you need

- **Python 3.11 or newer** and **Git**
- **An NVIDIA graphics card** is recommended. The app also runs without one (on the regular processor) but more slowly, and training without one is not practical.

## Use the app

**1. Download the project**

```bash
git clone https://github.com/Oxxidation/GriffProject.git
cd GriffProject
```

**2. Make a Python environment for it.** This keeps the project's packages separate from everything else on your computer.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal line. Run this activate line again whenever you open a new terminal for this project.

**3. Install PyTorch.** Go to https://pytorch.org/get-started/locally/, pick your system, and run the command it shows. With an NVIDIA card, pick a CUDA version (RTX 50-series cards need CUDA 12.8 or newer). Without one, pick "CPU".

**4. Install everything else**

```bash
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
$env:PYYAML_FORCE_LIBYAML="0"; pip install pyyaml
```

Mac or Linux:

```bash
PYYAML_FORCE_LIBYAML=0 pip install pyyaml
```

**5. Start the app**

```bash
python app/app.py
```

When it prints `Running on local URL: http://127.0.0.1:7860`, open that address in your browser.

- **Photo** tab: upload a picture, pick a style, and drag the sliders.
- **Webcam** tab: allow camera access, then press record to see yourself as a painting.
- **Speed comparison** tab: runs the old, slow method next to this model. The first time you use it, it downloads the VGG16 network (about 530 MB).

Press `Ctrl+C` in the terminal to stop the app.

## Do it yourself

Follow "Use the app" steps 1 to 4 first. Then run these in order, from the project folder, with the environment activated.

**Step 1. Check your graphics card**

```bash
python scripts/check_gpu.py
```

It should end with `PASS`. If it says CUDA is not available, reinstall PyTorch with a CUDA version (step 3 above).

**Step 2. Download the practice photos** (about 780 MB, a few minutes)

```bash
python scripts/download_coco.py
```

This puts 5,000 everyday photos in `data/coco/val2017/`. It's safe to run again; it skips the download if the photos are already there.

**Step 3. Run the tests** (a few seconds)

```bash
python -m pytest
```

These check that the scoring math, the network, and the painting code all work. It should say `14 passed`.

**Step 4. Tell Python where the code is.** Do this once per terminal.

Windows (PowerShell):

```powershell
$env:PYTHONPATH="src"
```

Mac or Linux:

```bash
export PYTHONPATH=src
```

**Step 5. Do a 1-minute trial run**

```bash
python -m styletransfer.train --iterations 1000 --out runs/trial
```

This makes sure training works and that the settings are sensible. The first time, it also downloads the VGG16 judge network (about 530 MB). Every 100 rounds it prints a line like this:

```
it    100  content 924558.00  style 3050681.50  tv    0.03    47.3 img/s  eta   2.5 min
```

`content` and `style` are the judge's scores (lower is better), and `eta` is the minutes left. When it finishes, open `runs/trial/samples/`: you'll see one test photo painted in all four styles. They'll be rough after only 1,000 rounds, but the four should already look different.

**Step 6. Train the real model** (about 30 minutes on an RTX 5080, longer on smaller cards)

```bash
python -m styletransfer.train --out runs/my_model
```

This runs 40,000 rounds with the same settings as the model in `weights/final.pth`. Leave the terminal open while it runs. A new picture appears in `runs/my_model/samples/` every 1,000 rounds, so you can watch it improve.

**Step 7. Try your model on a photo**

```bash
python -m styletransfer.infer --checkpoint runs/my_model/checkpoint.pth --input data/coco/val2017/000000559707.jpg --style starry_night
```

Swap in any photo and any style: `starry_night`, `great_wave`, `the_scream`, or `composition_vii`. Add `--blend-with great_wave --alpha 0.5` to mix two styles. The result is saved in `runs/infer/`.

**Step 8. Use your model in the app**

```bash
python app/app.py --checkpoint runs/my_model/checkpoint.pth
```

### Experiments to try

- **Stronger or milder style.** Change the style weight (the default is `3e10`; the first model used `1e10`, which came out milder):

  ```bash
  python -m styletransfer.train --style-weight 1e10 --out runs/milder
  ```

- **Your own painting.** Put a `.jpg` in `styles/`, add it to the `styles:` list in `configs/default.yaml` with a short name, and train again (step 6).

- **The old, slow method.** Paint one photo by trial and error, with no trained network:

  ```bash
  python -m styletransfer.gatys --content data/coco/val2017/000000559707.jpg --style starry_night
  ```

## Layout of my project

```
app/app.py               the web app
configs/default.yaml     all training settings
styles/                  the four paintings + SOURCES.md
src/styletransfer/       data, vgg (the judge), losses, model (the painter), train, infer, gatys
scripts/                 check_gpu.py, download_coco.py
tests/                   automatic checks
weights/                 the trained model (final.pth) and its settings
```
