import itertools
import math
from pathlib import Path
import numpy as np
from PIL import Image
import random

if __name__ == "__main__":
  from generate import generate_random_bytes, iter_bits_from_bytes
  from ColorModel import ColorModel
else:
  from .generate import generate_random_bytes, iter_bits_from_bytes
  from .ColorModel import ColorModel


class LsbMatching:

  def __init__(self):
    """Initializes the LsbMatching class."""
    self.header_len_bits = 32  # 32 bits to store the message length in bytes


  no_channels = {
    ColorModel.GRAYSCALE: 1, 
    ColorModel.RGB: 3
  }


  def encode_message(self, 
    image_path: Path, 
    output_path: Path,
    target_threshold: float,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    if color_model not in list(ColorModel):
      raise ValueError(f"Unsupported image mode: {color_model}. Supported modes: {list(ColorModel)}")
    
    if not Path(image_path).exists() or not Path(image_path).is_file():
      raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not output_path.parent.exists():
      raise FileNotFoundError("Stego directory does not exist.")
    
    # Image processing
    img = Image.open(image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.array(img, dtype=np.uint8)
    channel_values = pixel_array.flatten()

    # Capacity
    embedding_total_capacity = channel_values.size
    bits_to_encode = math.floor(target_threshold * embedding_total_capacity)
    payload_length_bits = bits_to_encode - self.header_len_bits

    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label6 = "Header size:".ljust(40)
      label2 = "Total capacity:".ljust(40)
      label4 = "Target threshold:".ljust(40)
      label3 = "Net capacity (excluding header):".ljust(40)

      print()
      print(f"{label1} {width}x{height}")
      print(f"{label6} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label2} {embedding_total_capacity:n} bits ({embedding_total_capacity / 8:n} bytes)")
      print(f"{label4} {target_threshold * 100}%")
      print(f"{label3} {payload_length_bits:n} bits ({payload_length_bits / 8:n} bytes)")

    # Generate message and embedding path
    message_in_bytes = generate_random_bytes(payload_length_bits)
    header_in_bytes = self._generate_header_bytes(payload_length_bits)

    # Embedd
    bitstream = itertools.chain(
      iter_bits_from_bytes(header_in_bytes, self.header_len_bits),
      iter_bits_from_bytes(message_in_bytes, payload_length_bits)
    )

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Gracescale is currently implemented")
    
    for i, (bit, indivudial_c_val) in enumerate(zip(bitstream, channel_values)):
      current_lsb = indivudial_c_val & 1
      
      if bit == current_lsb:
        continue

      new_channel_val = int(indivudial_c_val) + random.choice([-1, 1])
      if new_channel_val > 255:
        new_channel_val = indivudial_c_val - 1
      elif new_channel_val < 0:
        new_channel_val = indivudial_c_val + 1

      channel_values[i] = np.uint8(new_channel_val)

    # Reprocess pixels and save
    pixel_array = channel_values.reshape(height, width, self.no_channels[color_model])
    if self.no_channels[color_model] == 1:
      pixel_array = pixel_array.squeeze()
    Image.fromarray(pixel_array, mode=color_model.value).save(output_path)

    # Returning message for testing
    return message_in_bytes


  def _generate_header_bytes(self, payload_len_bits: int) -> bytes:
    if payload_len_bits < 0 or payload_len_bits >= 2**self.header_len_bits:
      raise ValueError("Payload too large to fit in header.")

    header_len_bytes = self.header_len_bits // 8  # 4 bytes
    return payload_len_bits.to_bytes(header_len_bytes, byteorder="big")


  def decode_message(self, 
    stego_image_path: Path,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    
    if not Path(stego_image_path).exists() or not Path(stego_image_path).is_file():
      raise FileNotFoundError(f"Stego image file not found: {stego_image_path}")
    
    # Image processing
    img = Image.open(stego_image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.array(img, dtype=np.uint8)
    channel_values = pixel_array.flatten()
    
    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")
    
    # Extract header to determine payload length
    header_bits = []
    for i in range(self.header_len_bits):
      lsb = channel_values[i] & 1
      header_bits.append(lsb)
    
    # Convert header bits to bytes (MSB-first within each byte)
    header_bytes = bytearray()
    for i in range(0, self.header_len_bits, 8):
      byte_bits = header_bits[i:i+8]
      byte_value = 0
      for bit in byte_bits:
        byte_value = (byte_value << 1) | bit  # MSB first
      header_bytes.append(byte_value)
    
    # Convert bytes to integer (big-endian, matching the encoding)
    payload_length_bits = int.from_bytes(header_bytes, byteorder="big")
    
    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label2 = "Header size:".ljust(40)
      label3 = "Payload length (from header):".ljust(40)
      
      print()
      print(f"{label1} {width}x{height}")
      print(f"{label2} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label3} {payload_length_bits:n} bits ({payload_length_bits / 8:n} bytes)")
    
    # Extract message bits
    message_bits = []
    start_index = self.header_len_bits
    end_index = start_index + payload_length_bits
    
    for i in range(start_index, end_index):
      lsb = channel_values[i] & 1
      message_bits.append(lsb)
    
    # Convert bits to bytes (MSB-first within each byte)
    message_bytes = bytearray()
    for i in range(0, len(message_bits), 8):
      byte_bits = message_bits[i:i+8]
      byte_value = 0
      for bit in byte_bits:
        byte_value = (byte_value << 1) | bit  # MSB first
      message_bytes.append(byte_value)
    
    return bytes(message_bytes)


if __name__ == "__main__":
  input_path = Path('./Picture1.png')
  output_image = Path('./output_img.png')
  target_threshold = 0.5
 
  steg_tool = LsbMatching()

  print("\n...encoding random data")
  generated_message = steg_tool.encode_message(
    image_path=input_path, 
    output_path=output_image, 
    target_threshold= target_threshold, 
    verbose=True
  )

  print("\n...decoding image")
  decoded_message = steg_tool.decode_message(
    stego_image_path=output_image,
    verbose=True
  )

  print(f"\ngenerated msg lenght: {len(generated_message)}\ndecoded msg lenght: {len(decoded_message)}")
  print(f"equal: {generated_message == decoded_message}")
  print(f"First 8 bytes generated: {generated_message[:8]}")
  print(f"First 8 bytes decoded: {decoded_message[:8]}")
  print(f"Last 8 bytes generated: {generated_message[-8:]}")
  print(f"Last 8 bytes decoded: {decoded_message[-8:]}")
