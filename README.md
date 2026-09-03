# Road Accident Detection System

A computer vision system that watches road footage or images and flags accidents automatically. It was built for an IEEE project focused on road safety, using a YOLOv8 model trained to recognize moderate and severe crashes.

## What it does

Feed it a video or a set of images and it will scan every frame for signs of an accident. When it finds one with high enough confidence, it starts tracking that spot, waits a short amount of time to confirm it's not a false alarm, then logs it as a confirmed accident with a timestamp. Every confirmed detection gets written to an Excel log (`accident_log.xlsx`) along with a running global count, so you always know how many accidents have been caught across every file you've processed.

There's both a GUI and a command line mode:

* The GUI (`accident_detection_gui.py`) lets you load files through a file picker, tweak the confidence and persistence thresholds with sliders, watch detections happen live on screen, and jump between files.
* The CLI mode (in `accident_detection.py`) does the same detection and logging without the interface, useful if you just want to process a batch of files quickly or run it on a server without a display.

## How it works

Detection is handled by a YOLOv8 model (via the `ultralytics` package) that classifies frames into `moderate` and `severe` accident categories. When the model spots something above the confidence threshold, the app doesn't count it immediately. Instead it starts an OpenCV tracker (KCF, with a CSRT fallback) on that region and only logs it as a real accident once the detection has held steady for a few seconds. This persistence check is what keeps a single flickering false positive from being logged as ten separate accidents.

Every confirmed accident gets a marker drawn on screen, a row appended to the Excel log, and a bump to the running total, which is also saved to `accident_count.txt` so the counter survives between sessions.

## Tech used

* Python
* Ultralytics YOLOv8 for object detection
* OpenCV (including its tracking API) for video processing and on-screen overlays
* cvzone for drawing bounding boxes and labels
* Tkinter and Pillow for the GUI
* openpyxl for writing the Excel log

## Installation

Install the dependencies:

```bash
pip install ultralytics opencv-python opencv-contrib-python cvzone openpyxl pillow numpy
```

You'll also need a trained YOLOv8 weights file named `best.pt` in the project root. The weights and sample media used during development aren't included in this repo since they're large binary files, so you'll need to either train your own model on an accident dataset or supply your own `best.pt`.

## Running it

To launch the GUI:

```bash
python accident_detection.py --gui
```

To run from the command line against your own files:

```bash
python accident_detection.py --files path/to/video.mp4 path/to/image.jpg --confidence 0.7 --persistence 2.0
```

`--confidence` controls how sure the model needs to be before it flags something, and `--persistence` controls how many seconds a detection has to hold before it counts as a confirmed accident.
