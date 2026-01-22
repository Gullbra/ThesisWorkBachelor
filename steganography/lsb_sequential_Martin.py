
from pathlib import Path
from PIL import Image
import sys
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, '')
if __name__ == "__main__":
  from generate import generate_random_bytes
else:
  from .generate import generate_random_bytes


class LsbSequential:
  def __init__(self):
    pass

  no_channels = {
    'L': 1, 
    'RGB': 3
  }

  def encode_message(self, 
    image_path: Path, 
    target_threshold: float, 
    output_path: Path, 
    verbose: bool = False,
    mode: str = 'L',
    delimiter: str = "###END###"
  ) -> bytes:
    """
    Encode random data into an image using LSB steganography.
    """

    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    if mode not in self.no_channels:
      raise ValueError(f"Unsupported image mode: {mode}. Supported modes: {list(self.no_channels.keys())}")

    # Open image as mode and convert to numpy array
    img = Image.open(image_path).convert(mode)
    width, height = img.size
    img_array = np.array(img)
    
    # Storage capacity
    capacity_total_bits = width * height * self.no_channels[mode]
    delimiter_bits = len(delimiter) * 8
    capacity_netto_bits = capacity_total_bits - delimiter_bits
    if capacity_netto_bits <= 0:
      raise ValueError("Image too small to encode delimiter.")

    no_of_bits_to_encode = int(target_threshold * capacity_netto_bits)

    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label2 = "Total capacity:".ljust(40)
      label3 = "Net capacity (excluding delimiter):".ljust(40)
      label4 = "Target threshold:".ljust(40)
      label5 = "Bits to encode:".ljust(40)
      label6 = "Delimiter size:".ljust(40)

      print()
      print(f"{label1} {width}x{height}")
      print(f"{label6} {delimiter_bits:n} bits ({delimiter_bits / 8:n} bytes)")
      print(f"{label2} {capacity_total_bits:n} bits ({capacity_total_bits / 8:n} bytes)")
      print(f"{label3} {capacity_netto_bits:n} bits ({capacity_netto_bits / 8:n} bytes)")
      print(f"{label4} {target_threshold * 100}%")
      print(f"{label5} {no_of_bits_to_encode:n} bits ({no_of_bits_to_encode / 8:n} bytes)")

    # Generate bytes to encode
    secret_message = generate_random_bytes(no_of_bits_to_encode)
    delimiter_bytes = delimiter.encode('ascii')

    if verbose:
      label1 = "Generated message length:".ljust(40)
      label2 = "Converted delimiter length:".ljust(40)
      print(f"{label1} {len(secret_message):n} bytes")
      print(f"{label2} {len(delimiter_bytes):n} bytes")

    full_message_in_bytes = secret_message + delimiter_bytes
    total_message_bits = no_of_bits_to_encode + delimiter_bits
    
    # Pre-compute all bits to encode as numpy array
    all_bits = np.zeros(total_message_bits, dtype=np.uint8)
    bit_position = 0
    
    # Calculate starting bit index for the first byte
    first_byte_bits = total_message_bits % 8
    if first_byte_bits == 0:
      first_byte_bits = 8
    
    # Extract bits from all bytes
    for byte_idx, byte_val in enumerate(full_message_in_bytes):
      if byte_idx == 0:
        # First byte: extract only the relevant bits
        for i in range(first_byte_bits):
          all_bits[bit_position] = (byte_val >> (first_byte_bits - 1 - i)) & 1
          bit_position += 1
      else:
        # Subsequent bytes: extract all 8 bits
        for i in range(8):
          all_bits[bit_position] = (byte_val >> (7 - i)) & 1
          bit_position += 1
    
    # Flatten image array to 1D for faster processing
    if mode == 'RGB':
      flat_pixels = img_array.reshape(-1, 3)
    else:  # mode == 'L'
      flat_pixels = img_array.reshape(-1, 1)
    
    # Calculate how many pixels we need to modify
    pixels_needed = (total_message_bits + self.no_channels[mode] - 1) // self.no_channels[mode]
    
    # Clear LSBs and set new bits using vectorized operations
    flat_pixels_copy = flat_pixels.copy()
    flat_pixels_copy[:pixels_needed] &= 0xFE  # Clear LSBs
    
    # Set the message bits
    bit_idx = 0
    for pixel_idx in range(pixels_needed):
      for channel_idx in range(self.no_channels[mode]):
        if bit_idx < total_message_bits:
          flat_pixels_copy[pixel_idx, channel_idx] |= all_bits[bit_idx]
          bit_idx += 1
    
    # Reshape back to image dimensions
    if mode == 'RGB':
      modified_array = flat_pixels_copy.reshape(height, width, 3)
    else:
      modified_array = flat_pixels_copy.reshape(height, width)
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(modified_array, mode=mode)
    result_img.save(output_path, 'PNG')
    
    if verbose:
      print(f"Random data encoded successfully!")
    print(f"Stego image saved to {output_path.resolve()}")

    return secret_message


  def decode_message(self, image_path: Path, delimiter: str = "###END###", mode: str = 'L') -> bytes:
    """
    Decode a message from an image using LSB steganography.
    """
    # Open image and convert to numpy array
    img = Image.open(image_path).convert(mode)
    img_array = np.array(img)
    
    # Flatten image array
    if mode == 'RGB':
      flat_pixels = img_array.reshape(-1, 3)
    else:
      flat_pixels = img_array.reshape(-1, 1)
    
    # Extract all LSBs using vectorized operation
    all_lsbs = (flat_pixels & 1).flatten()
    
    # Convert delimiter to bit list
    delimiter_bytes = delimiter.encode('ascii')
    delimiter_bits = np.zeros(len(delimiter_bytes) * 8, dtype=np.uint8)
    bit_idx = 0
    for byte in delimiter_bytes:
      for i in range(7, -1, -1):
        delimiter_bits[bit_idx] = (byte >> i) & 1
        bit_idx += 1
    
    delimiter_length = len(delimiter_bits)
    
    # Search for delimiter using numpy operations
    found_at = -1
    max_search = min(len(all_lsbs) - delimiter_length + 1, len(all_lsbs))
    
    for i in range(max_search):
      if np.array_equal(all_lsbs[i:i+delimiter_length], delimiter_bits):
        found_at = i
        break
    
    if found_at == -1:
      raise ValueError("No message found or message corrupted.")
    
    # Extract message bits (before delimiter)
    message_bits = all_lsbs[:found_at]
    
    if len(message_bits) == 0:
      return b''
    
    # Convert bits to bytes
    message_bytes = bytearray()
    
    # Handle first potentially incomplete byte
    first_byte_bits = len(message_bits) % 8
    if first_byte_bits > 0:
      byte_val = 0
      for bit in message_bits[:first_byte_bits]:
        byte_val = (byte_val << 1) | bit
      message_bytes.append(byte_val)
      start_index = first_byte_bits
    else:
      start_index = 0
    
    # Handle remaining complete bytes
    for i in range(start_index, len(message_bits), 8):
      byte_bits = message_bits[i:i+8]
      byte_val = 0
      for bit in byte_bits:
        byte_val = (byte_val << 1) | bit
      message_bytes.append(byte_val)
    
    return bytes(message_bytes)


if __name__ == "__main__":
  steg_tool = LsbSequential()

  input_path = Path('./Picture1.png')
  target_threshold = 0.5
  output_image = Path('./output_img.png')
  delimiter = "###END###"

  try:
    print("\n...encoding random data")
    generated_message = steg_tool.encode_message(
      input_path, 
      target_threshold, 
      output_image,
      verbose=True,
      delimiter=delimiter
    )
    print(f"Generated data length: {len(generated_message)} bytes")

    print("\n...decoding message")
    hidden_data = steg_tool.decode_message(output_image, delimiter=delimiter)
    print(f"Decoded data length: {len(hidden_data)} bytes")
    
    print(f"Encoding-Decoding successful: {generated_message == hidden_data}")
  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    