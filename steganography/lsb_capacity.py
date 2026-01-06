from pathlib import Path
from PIL import Image
import sys
from steganography.steganography import Steganography
from .generate import generate_random_bytes


class LsbSequential(Steganography):
  def __init__(self):
    pass


  def encode_message(self, 
    image_path: Path, 
    threshold: float, 
    output_path: Path, 
    verbose: bool = False,
    delimiter: str = "###END###"
  ) -> None:
    """
    Encode random data into an image using LSB steganography.
    
    Args:
      image_path: Path to the input image
      threshold: Float between 0 and 1 representing percentage of capacity to use
      output_path: Path to save the output image
    """
    # Open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Get image dimensions
    width, height = img.size
    pixels = img.load()
    
    # Storage capacity vars
    total_bits = width * height * 3
    netto_capacity = total_bits - len(delimiter) * 8
    bits_to_encode = int(threshold * netto_capacity)
    
    # Generate random bytes
    secret_message = generate_random_bytes(bits_to_encode)
    
    # Convert bytes to binary string
    binary_message = ''.join(format(byte, '08b') for byte in secret_message)
    
    # Handle potentially masked MSB - if num_bits wasn't multiple of 8,
    # the first byte may have leading zeros that are significant
    if bits_to_encode % 8 != 0:
      # Adjust the first byte to only use the required number of bits
      first_byte_bits = bits_to_encode % 8
      binary_message = format(secret_message[0], '08b')[-first_byte_bits:] + \
                      ''.join(format(byte, '08b') for byte in secret_message[1:])
    
    # Add delimiter at the end
    binary_delimiter = ''.join(format(ord(char), '08b') for char in delimiter)
    binary_message = binary_message + binary_delimiter
    
    # Verify message fits
    if len(binary_message) > total_bits:
      raise ValueError(f"Message too large. Required: {len(binary_message)} bits, Available: {total_bits} bits")
    
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
    output_file = output_path / image_path.name
    img.save(output_file, 'PNG')
    if verbose:
      print(f"Random data encoded successfully!")
      print(f"Net capacity: {netto_capacity} bits ({netto_capacity // 8} bytes)")
      print(f"Encoded data: {bits_to_encode} bits ({bits_to_encode / 8:.2f} bytes)")
      print(f"Threshold used: {threshold * 100}%")
      print(f"Output saved to: {output_file}")
    else: 
      print(f"Stego image saved to {output_file}")


  def decode_message(self, image_path):
    """
    Decode a message from an image using LSB steganography.
    
    Args:
      image_path: Path to the image containing hidden message
      
    Returns:
      The decoded message as bytes (before delimiter)
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
    
    # Convert binary to bytes
    message_bytes = bytearray()
    for i in range(0, len(binary_message), 8):
      byte = binary_message[i:i+8]
      if len(byte) == 8:
        byte_val = int(byte, 2)
        message_bytes.append(byte_val)
        
        # Check for end delimiter in the accumulated bytes
        if len(message_bytes) >= 9:
          try:
            last_9 = bytes(message_bytes[-9:]).decode('ascii')
            if last_9 == "###END###":
              # Return message without delimiter
              return bytes(message_bytes[:-9])
          except:
            pass
    
    return f"No message found or message corrupted.".encode()


if __name__ == "__main__":
  steg_tool = LsbSequential()

  input_path = './Picture1.png'
  threshold = 0.5  # Use 50% of available capacity
  output_image = './output_img.png'

  try:
    print("\n...encoding random data")
    steg_tool.encode_message(input_path, threshold, output_image)

    print("\n...decoding message")
    hidden_data = steg_tool.decode_message(output_image)
    print(f"Decoded data length: {len(hidden_data)} bytes")
    print(f"First 16 bytes (hex): {hidden_data[:16].hex()}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
