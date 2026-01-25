
import itertools
import math
from pathlib import Path
import numpy as np
from PIL import Image
import random
from enum import Enum
import locale

locale.setlocale(locale.LC_ALL, '')
if __name__ == "__main__":
  from generate import generate_random_bytes, iter_bits_from_bytes

else:
  from .generate import generate_random_bytes, iter_bits_from_bytes



class ColorModel(str, Enum):
  RGB = 'RGB'
  GRAYSCALE = 'L'



class LsbRandom:

  def __init__(self, key):
    if not isinstance(key, int):
      raise ValueError("Key must be an integer (PRNG seed).")
      
    self.key = key
    self.header_len_bits = 32


  no_channels = {
    ColorModel.GRAYSCALE: 1, 
    ColorModel.RGB: 3
  }
  bits_per_channel = {
    ColorModel.GRAYSCALE: [1], 
    ColorModel.RGB: [3, 3, 2]
  }
  pixel_to_channel_map = {
    ColorModel.GRAYSCALE: lambda x: [x], 
    ColorModel.RGB: lambda x: list(x)
  }


  def encode_message(self, 
    image_path: Path, 
    output_path: Path,
    target_threshold: float, 
    color_model: str = ColorModel.GRAYSCALE,
    verbose = False,
  ) -> bytes:
    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    if color_model not in ColorModel:
      raise ValueError(f"Unsupported image mode: {color_model}. Supported modes: {list(self.no_channels.keys())}")
    
    # Image Processing
    img = Image.open(image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.array(img, dtype=np.uint8)
    channel_values = pixel_array.flatten()

    # Capacity
    embedding_bits_per_pixel = sum(self.bits_per_channel[color_model])
    embedding_total_capacity = channel_values.size * embedding_bits_per_pixel
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
    embedding_path = self._get_shuffled_indices(len(channel_values))

    # Embedd
    bitstream = itertools.chain(
      iter_bits_from_bytes(header_in_bytes, self.header_len_bits),
      iter_bits_from_bytes(message_in_bytes, payload_length_bits)
    )

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Gracescale is currently implemented")

    # temp_bits = []
    # bit_in_channel = 0;
    path_index = 0;
    for bit in bitstream:
      cindex = embedding_path[path_index]
      channel_values[cindex] = (channel_values[cindex] & 0xFE) | bit
      path_index += 1

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


  def _get_shuffled_indices(self, len_channel_values) -> list[int]:
    indices = list(range(len_channel_values))
    random.Random(self.key).shuffle(indices)
    return indices
  

  def decode_message(self, 
    stego_image_path: Path, 
    color_model: str = ColorModel.GRAYSCALE,
    verbose = False,
  ) -> bytes:
    if color_model not in ColorModel:
      raise ValueError(f"Unsupported image mode: {color_model}. Supported modes: {list(self.no_channels.keys())}")
    
    # Image Processing
    img = Image.open(stego_image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.array(img, dtype=np.uint8)
    channel_values = pixel_array.flatten()

    # Get the same shuffled indices used during encoding
    embedding_path = self._get_shuffled_indices(len(channel_values))

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")

    # Extract header to determine payload length
    header_bits = []
    for i in range(self.header_len_bits):
      cindex = embedding_path[i]
      bit = channel_values[cindex] & 1
      header_bits.append(bit)
    
    # Convert header bits to payload length
    header_bytes = self._bits_to_bytes(header_bits)
    payload_len_bits = int.from_bytes(header_bytes, byteorder="big")

    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label2 = "Header size:".ljust(40)
      label3 = "Payload size:".ljust(40)
      
      print()
      print(f"{label1} {width}x{height}")
      print(f"{label2} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label3} {payload_len_bits:n} bits ({payload_len_bits / 8:n} bytes)")

    # Extract payload bits
    payload_bits = []
    for i in range(self.header_len_bits, self.header_len_bits + payload_len_bits):
      cindex = embedding_path[i]
      bit = channel_values[cindex] & 1
      payload_bits.append(bit)
    
    # Convert payload bits to bytes
    message_bytes = self._bits_to_bytes(payload_bits)
    
    return message_bytes


  def _bits_to_bytes(self, bits: list[int]) -> bytes:
    """Convert a list of bits to bytes."""
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
      byte_bits = bits[i:i+8]
      # Pad with zeros if the last byte is incomplete
      while len(byte_bits) < 8:
        byte_bits.append(0)
      # Convert 8 bits to a byte (MSB first)
      byte_value = 0
      for bit in byte_bits:
        byte_value = (byte_value << 1) | bit
      byte_array.append(byte_value)
    
    return bytes(byte_array)

if __name__ == "__main__":
  steg_tool = LsbRandom(123456789)

  input_path = Path('./Picture1.png')
  target_threshold = 0.5
  output_image = Path('./output_img.png')
 
  print("\n...encoding random data")
  generated_message = steg_tool.encode_message(
    image_path=input_path, 
    target_threshold= target_threshold, 
    output_path=output_image, 
    verbose=True
  )

  print("\n...decoding image")
  decoded_message = steg_tool.decode_message(
    stego_image_path=output_image,
    verbose=True
  )

  print(f"generated msg lenght: {len(generated_message)}\ndecoded msg lenght: {len(decoded_message)}")
  print(f"equal: {generated_message == decoded_message}")

  print(generated_message[:8])
  print(decoded_message[:8])

