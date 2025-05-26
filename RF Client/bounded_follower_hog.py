import cv2
import numpy as np
from bounded_follower import BoundedFollower

class BoundedFollowerHog(BoundedFollower):
    """
    A follower that detects a person in the image using HOG,
    draws a bounding box around them, and displays the result.
    """
    def __init__(self, min_bound=0.5, max_bound=0.8):
        """
        Initialize the HOG descriptor/person detector.
        
        Args:
            min_bound: Minimum bound as a percentage of image size (0.0 to 1.0)
            max_bound: Maximum bound as a percentage of image size (0.0 to 1.0)
        """
        # Call the parent class constructor
        super().__init__(min_bound, max_bound)
        
        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Parameters for detection
        self.scale_factor = 1.05  # Smaller scale factor for better performance
        self.win_stride = (4, 4)  # Smaller stride for better performance
        self.padding = (8, 8)
        
    def processImage(self, image_data):
        """
        Detect a person in the image, draw a bounding box, and display the result.
        
        Args:
            image_data: The raw image data to process (numpy array of bytes)
            
        Returns:
            str: A message indicating the detection result
        """
        # Decode the image data
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return "Failed to decode image"
        
        # Resize image for better performance (smaller image = faster processing)
        height, width = image.shape[:2]
        max_dimension = 400  # Limit the maximum dimension to 400 pixels
        
        # Only resize if the image is larger than max_dimension
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
        else:
            resized_image = image
            
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Detect people in the image
        boxes, weights = self.hog.detectMultiScale(
            gray, 
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale_factor
        )
        
        # Create a copy of the original image to draw on
        result_image = image.copy()
        
        # Draw bounding boxes around detected people
        person_count = 0
        for (x, y, w, h) in boxes:
            # If we resized the image, scale the bounding box back to original size
            if max(height, width) > max_dimension:
                scale_back = max(height, width) / max_dimension
                x = int(x * scale_back)
                y = int(y * scale_back)
                w = int(w * scale_back)
                h = int(h * scale_back)
            else:
                # Convert to integers
                x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw the bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            person_count += 1
        
        # Draw the boundary rectangles
        self.draw_bounds(result_image, width, height)
        
        # Display the result
        cv2.imshow("Bounded Follower", result_image)
        
        # Return a message with the detection result
        if person_count > 0:
            return f"Detected {person_count} person(s) in the image"
        else:
            return "No persons detected in the image"
