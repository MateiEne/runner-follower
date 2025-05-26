# Follower Simulator Client

This is a client application that connects to a server, receives images, and processes them using different person detection methods.

## Setup

### 1. Download YOLOv4-tiny weights

To use the YOLOv4-tiny person detector, you need to download the pre-trained weights file:

```bash
# Download YOLOv4-tiny weights (approximately 23MB)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

Or download it manually from this URL:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Save the file in the same directory as the Python scripts.

### 2. Install required packages

```bash
pip install opencv-python numpy
```

## Usage

The application supports two different person detection methods and allows customizing the boundary rectangles.

### Command-line Arguments

```bash
python main.py [--detector {hog,yolov4}] [--min-bound MIN_BOUND] [--max-bound MAX_BOUND]
```

- `--detector`: Person detector to use (hog or yolov4, default: yolov4)
- `--min-bound`: Minimum bound as a percentage of image size (0.0 to 1.0, default: 0.6)
- `--max-bound`: Maximum bound as a percentage of image size (0.0 to 1.0, default: 0.8)

### 1. HOG (Histogram of Oriented Gradients)

This is a lightweight method that works well on resource-constrained devices like Raspberry Pi.

```bash
# Use HOG detector with default bounds
python main.py --detector hog

# Use HOG detector with custom bounds
python main.py --detector hog --min-bound 0.5 --max-bound 0.9
```

### 2. YOLOv4-tiny

This method provides better accuracy but requires more computational resources.

```bash
# Use YOLOv4-tiny detector with default bounds
python main.py --detector yolov4

# Use YOLOv4-tiny detector with custom bounds
python main.py --detector yolov4 --min-bound 0.4 --max-bound 0.7
```

For backward compatibility, you can also use the old command format:
```bash
python main.py hog
python main.py yolov4
```

## Controls

- Press `q` or `Esc` to exit the application

## Implementation Details

The application follows an object-oriented design with the following components:

- `Follower` interface: Defines the contract for image processing classes
- `BoundedFollower`: Base class for followers that draw boundary rectangles
- `BoundedFollowerHog`: Implements person detection using OpenCV's HOG detector
- `BoundedFollowerYoloV4`: Implements person detection using YOLOv4-tiny

Both implementations:
- Draw bounding boxes around detected persons
- Draw two rectangular bounds (minimum and maximum) as specified by the command-line arguments
- Display the result in a window
