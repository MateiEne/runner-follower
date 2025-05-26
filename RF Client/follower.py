from abc import ABC, abstractmethod

class Follower(ABC):
    """
    Interface for image processing followers.
    """
    @abstractmethod
    def processImage(self, image_array):
        """
        Process an image and return a string command.
        
        Args:
            image_array: The image to process (numpy array)
            
        Returns:
            str: A string command from processing the image
        """
        pass
