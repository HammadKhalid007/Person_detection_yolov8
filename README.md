# Object Tracking with YOLO and OpenCV

- Demonstrates object tracking using YOLO and OpenCV.
- Interactive selection of objects for tracking based on mouse clicks.
- Automatic removal of trackers after 10 seconds of lost detection.

## Overview

- Uses YOLOv8n, an ultralytics YOLO variant, for robust person detection in video streams.
- Allows users to interactively select individuals for tracking by clicking on detected persons.
- Utilizes OpenCV's built-in trackers to follow selected persons for up to 2 seconds after they are no longer detected.
- Automatically removes trackers if a person is not detected again within the timeout period.

## Requirements

Ensure you have Python 3.x installed. Install dependencies using the following command:

```bash
pip install -r requirements.txt