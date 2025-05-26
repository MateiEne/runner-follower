import cv2
from follower import Follower

class BoundedFollower(Follower):
    """
    A base class for followers that draw boundary rectangles.
    """
    def __init__(self, min_bound=0.5, max_bound=0.8, left_bound=0.4, right_bound=0.6):
        """
        Initialize the bounded follower.
        
        Args:
            min_bound: Minimum bound as a percentage of image size (0.0 to 1.0)
            max_bound: Maximum bound as a percentage of image size (0.0 to 1.0)
        """
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.left_bound = left_bound
        self.right_bound = right_bound
    
    def draw_bounds(self, image, width, height):
        """
        Draw the minimum and maximum boundary rectangles on the image.
        
        Args:
            image: The image to draw on
            width: The width of the image
            height: The height of the image
        """
        # Draw the minimum bound rectangle
        min_rect_width = int(width * self.min_bound)
        min_rect_height = int(height * self.min_bound)
        
        # Calculate the top-left and bottom-right coordinates to center the rectangle
        min_rect_x = int((width - min_rect_width) / 2)
        min_rect_y = int((height - min_rect_height) / 2)
        
        # Draw the minimum bound rectangle contour
        cv2.rectangle(image, (min_rect_x, min_rect_y), 
                     (min_rect_x + min_rect_width, min_rect_y + min_rect_height), 
                     (0, 255, 255), 2)  # Yellow color
        
        # Draw the maximum bound rectangle
        max_rect_width = int(width * self.max_bound)
        max_rect_height = int(height * self.max_bound)
        
        # Calculate the top-left and bottom-right coordinates to center the rectangle
        max_rect_x = int((width - max_rect_width) / 2)
        max_rect_y = int((height - max_rect_height) / 2)
        
        # Draw the maximum bound rectangle contour
        cv2.rectangle(image, (max_rect_x, max_rect_y), 
                     (max_rect_x + max_rect_width, max_rect_y + max_rect_height), 
                     (0, 0, 255), 2)  # Red color
