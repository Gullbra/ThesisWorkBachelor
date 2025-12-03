
class Steganography:
  def __init__(self):
    pass

  def encode_message(self, image_path, message, output_path):
    """
    Encode a message into an image using LSB steganography.
    
    Args:
      image_path: Path to the input image
      message: String message to hide
      output_path: Path to save the output image
    """
    raise NotImplementedError("This method should be overridden by subclasses.")
  
  def decode_message(self, image_path):
    """
    Decode a message from an image using LSB steganography.
    
    Args:
      image_path: Path to the input image
    """