# venv\Scripts\activate.bat

import socket
import numpy as np
import cv2
import struct
import sys
import argparse
from bounded_follower_hog import BoundedFollowerHog
from bounded_follower_yolov4 import BoundedFollowerYoloV4

print("Client started")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Follower Simulator Client')
parser.add_argument('--detector', type=str, default='yolov4', choices=['hog', 'yolov4'],
                    help='Person detector to use (hog or yolov4)')
parser.add_argument('--min-bound', type=float, default=0.5,
                    help='Minimum bound as a percentage of image size (0.0 to 1.0)')
parser.add_argument('--max-bound', type=float, default=0.8,
                    help='Maximum bound as a percentage of image size (0.0 to 1.0)')
args = parser.parse_args()

# For backward compatibility with the old command-line argument format
if len(sys.argv) > 1 and sys.argv[1].lower() in ['hog', 'yolov4']:
    args.detector = sys.argv[1].lower()

# Create the appropriate follower instance
if args.detector == "yolov4":
    print(f"Using YOLOv4-tiny for person detection (bounds: {args.min_bound}, {args.max_bound})")
    follower = BoundedFollowerYoloV4(min_bound=args.min_bound, max_bound=args.max_bound)
else:
    print(f"Using HOG for person detection (bounds: {args.min_bound}, {args.max_bound})")
    follower = BoundedFollowerHog(min_bound=args.min_bound, max_bound=args.max_bound)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 2737))

print("Connected to server")

while True:
    # Check for key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is the ASCII value for Escape key
        break

    # read the image from server
    size_data = client.recv(4)
    
    if not size_data:
        print("Connection closed by server")
        break

    size = struct.unpack("I", size_data)[0]

    image_data = b""
    while len(image_data) < size:
        packet = client.recv(size - len(image_data))
        if not packet:
            print("No more data received")
            break
        image_data += packet

    if len(image_data) != size:
        print(f"Error: expected {size} bytes but received {len(image_data)} bytes")
        continue

    image = np.frombuffer(image_data, dtype=np.uint8)

    # get the command from the follower
    command = follower.processImage(image)
    # print(command)

    # get the command from the follower
    command = follower.processImage(image)
    # print(command)
    
    # send the command to the server
    command = command.encode('utf-8')
    command_size = struct.pack("I", len(command))
    client.sendall(command_size)
    client.sendall(command)


print("Closing connection")
client.close()
cv2.destroyAllWindows()
