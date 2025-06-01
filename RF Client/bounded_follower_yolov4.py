import cv2
import numpy as np
import os

class BoundedFollowerYoloV4():
    """
    A follower that detects a person in the image using YOLOv4-tiny,
    draws a bounding box around them, and displays the result.
    """
    def __init__(self, 
                 weights_path="yolov4-tiny.weights", 
                 config_path="yolov4-tiny.cfg", 
                 classes_path="coco.names",
                 min_bound=0.5,
                 max_bound=0.8,
                 left_bound=0.4,
                 right_bound=0.6):
        """
        Initialize the YOLOv4-tiny detector.
        
        Args:
            weights_path: Path to the YOLOv4-tiny weights file
            config_path: Path to the YOLOv4-tiny configuration file
            classes_path: Path to the COCO class names file
            min_bound: Minimum bound as a percentage of image size (0.0 to 1.0)
            max_bound: Maximum bound as a percentage of image size (0.0 to 1.0)
        """
        
        self.weights_path = weights_path
        self.config_path = config_path
        self.classes_path = classes_path
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.left_bound = left_bound
        self.right_bound = right_bound
        
        # Check if the required files exist
        self.model_ready = self._check_files()
        
        if self.model_ready:
            # Load YOLOv4-tiny model
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Get output layer names
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Set backend and target (optional, for better performance)
            # Uncomment these lines if you want to use OpenCV's DNN with CUDA
            # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def _check_files(self):
        """Check if the required files exist."""
        files_exist = True
        missing_files = []
        
        if not os.path.exists(self.weights_path):
            missing_files.append(self.weights_path)
            files_exist = False
        
        if not os.path.exists(self.config_path):
            missing_files.append(self.config_path)
            files_exist = False
        
        if not os.path.exists(self.classes_path):
            missing_files.append(self.classes_path)
            files_exist = False
        
        if not files_exist:
            print("ERROR: The following files are missing:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nTo use YOLOv4-tiny, download the required files:")
            print("1. YOLOv4-tiny weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
            print("2. YOLOv4-tiny config: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg")
            print("3. COCO class names: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
            print("\nSave these files in the current directory or specify their paths when creating the BoundedFollowerYoloV4 instance.")
        
        return files_exist
        
    def processImage(self, image_data):
        """
        Detect a person in the image using YOLOv4-tiny, draw a bounding box, and display the result.
        
        Args:
            image_data: The raw image data to process (numpy array of bytes)
            
        Returns:
            str: A message indicating the detection result
        """
        # Check if the model is ready
        if not self.model_ready:
            return "ERROR: YOLOv4-tiny model files are missing. See console for details."
            
        # Decode the image data
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return "Failed to decode image"
        
        # Create a copy of the image to draw on
        result_image = image.copy()
        
        # Get image dimensions
        height, width, _ = image.shape # height=1024, width=1024
        
        # Prepare the image for YOLOv4-tiny
        # YOLOv4-tiny expects 416x416 images
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set the input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class ID 0 in COCO dataset) with confidence > 0.5
                if class_id == 0 and confidence > 0.5:
                    # YOLO returns normalized coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw bounding boxes for detected persons
        person_count = 0
        
        command = "None|None"

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Ensure coordinates are within image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                
                # Draw the bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate the height of the person in pixels
                person_height_px = h
                
                # Calculate the percentage of the person's height relative to the image height
                height_percentage = (person_height_px / height) * 100
                
                # Add a label with confidence and height information
                label = f"Person: {confidences[i]:.2f}, Height: {person_height_px}px ({height_percentage:.1f}%)"
                cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               
                person_count += 1

                command = self.check_bounds(result_image, width, height, x, y, w, h)
                break
        
        # Display the result
        cv2.imshow("YOLOv4 Follower", result_image)
        
        return command
    
    def check_bounds(self, result_image, width, height, x, y, w, h):
        horizontal_command = self.check_horizontal_bounds(result_image, width, height, x, w)
        vertical_command = self.check_vertical_bounds(result_image, width, height, y)

        # Combine horizontal and vertical commands
        return f"{horizontal_command}|{vertical_command}"
        
    def check_horizontal_bounds(self, result_image, width, height, x, w):
        # Draw a vertical line at the width * self.left_bound
        left_bound_x = int(width * self.left_bound)
        cv2.line(result_image, (left_bound_x, 0), (left_bound_x, height), (255, 120, 0), 2)

        # Draw a vertical line at the width * self.right_bound
        right_bound_x = int(width * self.right_bound)
        cv2.line(result_image, (right_bound_x, 0), (right_bound_x, height), (0, 120, 255), 2)
 
        # Get the point at the center of left and right bounds
        desired_position = int((left_bound_x + right_bound_x) / 2)
        current_position = x + int(w / 2)
        return f"distance#{desired_position - current_position}"
    
        
    def check_vertical_bounds(self, result_image, width, height, y):
        # Draw the minimum bound rectangle
        min_rect_height = int(height * self.min_bound)
        # Calculate the top-left and bottom-right coordinates to center the rectangle
        min_rect_y = int((height - min_rect_height) / 2)
        # Draw a horizontal line at the top of the minimum bound rectangle
        cv2.line(result_image, (0, min_rect_y), (width, min_rect_y), (0, 255, 255), 2)
        
        # Draw the maximum bound rectangle
        max_rect_height = int(height * self.max_bound)
        # Calculate the top-left and bottom-right coordinates to center the rectangle
        max_rect_y = int((height - max_rect_height) / 2)
        # Draw a horizontal line at the top of the maximum bound rectangle
        cv2.line(result_image, (0, max_rect_y), (width, max_rect_y), (0, 0, 255), 2)


        # Draw a horizontal line at the middle of the minimum and maximum bounds
        desired_position = int((min_rect_y + max_rect_y) / 2)
        print(f"Desired movement position: {desired_position}")
        # Draw a horizontal line at the top of the detected person
        cv2.line(result_image, (0, y), (width, y), (255, 0, 0), 2)
        # assign the top of the detected person to current_position
        current_position = y

        return f"distance#{desired_position - current_position}"