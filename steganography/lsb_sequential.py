from pathlib import Path
from PIL import Image
import sys
import locale
locale.setlocale(locale.LC_ALL, '')
if __name__ == "__main__":
  from generate import generate_random_bytes
else:
  from .generate import generate_random_bytes


class LsbSequential:
  def __init__(self):
    pass


  def encode_message(self, 
    image_path: Path, 
    target_threshold: float, 
    output_path: Path, 
    verbose: bool = False,
    delimiter: str = "###END###"
  ) -> bytes:
    """
    Encode random data into an image using LSB steganography.
    """

    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    # Open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = img.load()
    
    # Storage capacity
    capacity_total_bits = width * height * 3
    delimiter_bits = len(delimiter) * 8
    capacity_netto_bits = capacity_total_bits - len(delimiter) * 8
    if capacity_netto_bits <= 0:
      raise ValueError("Image too small to encode delimiter.")

    no_of_bits_to_encode = int(target_threshold * capacity_netto_bits)
    # no_of_bits_to_encode = 3 + 8

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
    
    # Encode message into image
    byte_index = 0
    bit_index = 7 - (len(full_message_in_bytes) * 8 - total_message_bits)
    break_loops = False

    def next_bit() -> None:
      nonlocal byte_index, bit_index, break_loops
      if bit_index == 0:
        if byte_index + 1 >= len(full_message_in_bytes):
          if verbose:
            print("\nbreak flag set in next_bit()")
          break_loops = True
          return
        byte_index += 1
        if verbose:
          print(f"\n{byte_index}: ", end=" ")
        bit_index = 7
      else:
        bit_index -= 1
    
    if verbose:
      print("0: ", end=" ")

    for y in range(height):
      for x in range(width):
        r, g, b = pixels[x, y]
        channels = [r, g, b]

        for channel_index in range(len(channels)):
          current_byte = full_message_in_bytes[byte_index]
          current_bit = (current_byte >> bit_index) & 1
          if verbose:
            print(current_bit, end=" ")
          channels[channel_index] = (channels[channel_index] & 0xFE) | current_bit
          next_bit()
          if break_loops:
            if verbose:
              print("next_bit break")
            break
        
        r, g, b = channels
        pixels[x, y] = (r, g, b)
        if break_loops:
          if verbose:
            print("x loop break")
          break
      
      if break_loops:
        if verbose:
          print("y loop break")
        break
    
    # Save as PNG to avoid lossy compression
    img.save(output_path, 'PNG')
    if verbose:
      print(f"Random data encoded successfully!")
      print(f"Stego image saved to {output_path.resolve()}")

    return secret_message


  def decode_message(self, image_path: Path, delimiter: str = "###END###") -> bytes:
    """
    Decode a message from an image using LSB steganography.
    """
    # Open image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = img.load()
    
    # Convert delimiter to bit list once
    delimiter_bytes = delimiter.encode('ascii')
    delimiter_bits = []
    for byte in delimiter_bytes:
      for i in range(7, -1, -1):  # MSB to LSB
        delimiter_bits.append((byte >> i) & 1)
    
    delimiter_length = len(delimiter_bits)
    
    # Extract binary data from LSBs
    binary_message = []
    found_delimiter = False
    
    for y in range(height):
      for x in range(width):
        channels = list(pixels[x, y])
        
        for channel in channels:
          binary_message.append(channel & 1)
          
          # Check for delimiter when we have enough bits
          if len(binary_message) >= delimiter_length:
            if binary_message[-delimiter_length:] == delimiter_bits:
              found_delimiter = True
              break
        
        if found_delimiter:
          break
      
      if found_delimiter:
        break
    
    if not found_delimiter:
      raise ValueError("No message found or message corrupted.")

    # Remove delimiter from the end
    binary_message = binary_message[:-delimiter_length]
    
    # Convert bits to bytes
    message_bytes = bytearray()
    
    # Handle the first (potentially incomplete) byte
    first_byte_bits = len(binary_message) % 8
    if first_byte_bits > 0:
      # Pad on the left with zeros to make it 8 bits
      byte_bits = [0] * (8 - first_byte_bits) + binary_message[:first_byte_bits]
      byte_val = 0
      for bit in byte_bits:
        byte_val = (byte_val << 1) | bit
      message_bytes.append(byte_val)
      start_index = first_byte_bits
    else:
      start_index = 0
    
    if len(binary_message) <= 8:
      return bytes(message_bytes)

    # Handle the remaining complete bytes
    for i in range(start_index, len(binary_message), 8):
      byte_bits = binary_message[i:i+8]
      
      # Convert bits to byte value (MSB first)
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
      delimiter=delimiter
    )
    print(f"Generated data length: {len(generated_message)} bytes")
    # print(generated_message)

    print("\n...decoding message")
    hidden_data = steg_tool.decode_message(output_image, delimiter=delimiter)
    print(f"Decoded data length: {len(hidden_data)} bytes")
    # print(hidden_data)
    
    print(f"Endcoding-Decoding successful: {generated_message == hidden_data}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
