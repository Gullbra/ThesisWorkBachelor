from PIL import Image
import sys
from steganography.steganography import Steganography


class LsbSequential(Steganography):
  def __init__(self):
    pass


  def encode_message(self, image_path, message, output_path):
    """
    Claude.ai Written

    Encode a message into an image using LSB steganography.
    
    Args:
      image_path: Path to the input image
      message: String message to hide
      output_path: Path to save the output image
    """
    # Open image and convert to RGB
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Get image dimensions
    width, height = img.size
    pixels = img.load()
    
    # Add delimiter to mark end of message
    message = message + "###END###"
    
    # Convert message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # Check if message fits in image
    # Unsure if this is completely correct. Have to chek later /Martin
    max_bytes = (width * height * 3) // 8
    if len(message) > max_bytes:
      raise ValueError(f"Message too large. Maximum characters: {max_bytes}")
    
    # Encode message into image
    message_index = 0
    
    for y in range(height):
      for x in range(width):
        if message_index < len(binary_message):
          r, g, b = pixels[x, y]
          
          # Modify LSB of red channel
          if message_index < len(binary_message):
            r = (r & 0xFE) | int(binary_message[message_index])
            message_index += 1
          
          # Modify LSB of green channel
          if message_index < len(binary_message):
            g = (g & 0xFE) | int(binary_message[message_index])
            message_index += 1
          
          # Modify LSB of blue channel
          if message_index < len(binary_message):
            b = (b & 0xFE) | int(binary_message[message_index])
            message_index += 1
          
          pixels[x, y] = (r, g, b)
        else:
          break
      
      if message_index >= len(binary_message):
        break
    
    # Save as PNG to avoid lossy compression
    img.save(output_path, 'PNG')
    print(f"Message encoded successfully!")
    print(f"Output saved to: {output_path}")

    if image_path.split('.')[-1].lower() == 'jpg':
      print(f"Note: Saved as PNG to preserve LSB data. JPG compression would destroy the hidden message.")


  def decode_message(self, image_path):
    """
    Claude.ai Written

    Decode a message from an image using LSB steganography.
    
    Args:
      image_path: Path to the image containing hidden message
      
    Returns:
      The decoded message string
    """
    # Open image
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Get image dimensions
    width, height = img.size
    pixels = img.load()
    
    # Extract binary data from LSBs
    binary_message = ""
    
    for y in range(height):
      for x in range(width):
        r, g, b = pixels[x, y]
        
        # Extract LSB from each channel
        binary_message += str(r & 1)
        binary_message += str(g & 1)
        binary_message += str(b & 1)
    
    # Convert binary to text
    message = ""
    for i in range(0, len(binary_message), 8):
      byte = binary_message[i:i+8]
      if len(byte) == 8:
        char = chr(int(byte, 2))
        message += char
        
        # Check for end delimiter
        if message.endswith("###END###"):
          message = message[:-9]  # Remove delimiter
          return message
    
    return f"No message found or message corrupted: {message}."  


if __name__ == "__main__":
  steg_tool = LsbSequential()


  input_path = './Picture1.png'
  secret_message = "Testing, testing"
  output_image = './output_img.png'

  try:
    print("\n...encoding message")
    steg_tool.encode_message(input_path, secret_message, output_image)

    print("\n...decoding message")
    hidden_message = steg_tool.decode_message(output_image)
    print(f"Decoded message: {hidden_message}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
