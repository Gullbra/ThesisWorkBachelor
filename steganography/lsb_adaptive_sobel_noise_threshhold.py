
print("Importing modules for LsbSobelEdge test...")
import itertools
from scipy import ndimage
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
  from ColorModel import ColorModel
else:
  from .generate import generate_random_bytes, iter_bits_from_bytes
  from .ColorModel import ColorModel
print("Modules imported.")


"""
Papers:

https://ieeexplore.ieee.org/abstract/document/11045847

https://reference-global.com/2/v2/download/pdf/10.2478/cait-2021-0032

https://arxiv.org/abs/1601.02076
"""

class LsbSobelEdge:

  def __init__(self, noice_threshhold: float):
    if not (0.0 <= noice_threshhold <= 1.0):
      raise ValueError("The noice_threshhold must be a float between 0 and 1 (inclusive).")
    
    self.noice_threshhold = noice_threshhold
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
    channel_values = pixel_array.flatten().copy()

    # Find noisy pixels
    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")
    
    noisy_pixel_indices, num_noisy_pixels = self._find_noisy_pixels(pixel_array, verbose)

    # Capacity
    embedding_total_capacity = num_noisy_pixels
    bits_to_encode = math.floor(target_threshold * embedding_total_capacity)
    payload_length_bits = bits_to_encode - self.header_len_bits

    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label6 = "Header size:".ljust(40)
      label2 = "Total capacity:".ljust(40)
      label4 = "Target threshold:".ljust(40)
      label3 = "Net capacity (excluding header):".ljust(40)
      label5 = "Bits to encode".ljust(40)

      print()
      print(f"{label1} {width}x{height} ({width*height:n} pixels)")
      print(f"{label6} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label2} {embedding_total_capacity:n} bits ({embedding_total_capacity / 8:n} bytes)")
      print(f"{label3} {embedding_total_capacity - self.header_len_bits:n} bits ({(embedding_total_capacity - self.header_len_bits) / 8:n} bytes)")
      print(f"{label4} {target_threshold * 100}%")
      print(f"{label5} {bits_to_encode:n} bits")

    if bits_to_encode > embedding_total_capacity:
      raise ValueError("Not enough noixy pixels to encode message")
    
    # Generate message
    message_in_bytes = generate_random_bytes(payload_length_bits)
    header_in_bytes = self._generate_header_bytes(payload_length_bits)

    # Encode
    bitstream = itertools.chain(
      iter_bits_from_bytes(header_in_bytes, self.header_len_bits),
      iter_bits_from_bytes(message_in_bytes, payload_length_bits)
    )

    indices_index = 0
    for bit in bitstream:
      value_index = noisy_pixel_indices[indices_index]
      channel_values[value_index] = (channel_values[value_index] & 0xFE) | bit
      indices_index += 1

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


  def _find_noisy_pixels(self, pixel_array: np.ndarray, verbose: bool) -> np.ndarray:

    # Excluding the least significant bit, so that we can decode and encode with the same noise finding method
    img_data = (pixel_array & 0xFE).astype(float)

    # Sobel matrix operations
    gradient_magnitude = np.hypot(
      ndimage.sobel(img_data, axis=1), 
      ndimage.sobel(img_data, axis=0)
    )

    # Scaling the threshhold with the max noise value in the pixel,
    # and converts the data into an array of indices for pixels where the noise value is equal or higher to the scaled threshhold
    threshold_scaled = self.noice_threshhold * np.max(gradient_magnitude)
    binary_mask = gradient_magnitude >= threshold_scaled
    noisy_pixel_indices = np.where(binary_mask.flatten())[0]

    if verbose:
      print(f"First 10 noisy pixel indices: {noisy_pixel_indices[:10]}")
      print(f"Last 10 noisy pixel indices: {noisy_pixel_indices[-10:]}")

    return noisy_pixel_indices, noisy_pixel_indices.size


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
    
    # Find noisy pixels (same as during encoding)
    noisy_pixel_indices, num_noisy_pixels = self._find_noisy_pixels(pixel_array, verbose)
    
    # Extract header bits from noisy pixels
    header_bits = []
    for i in range(self.header_len_bits):
      value_index = noisy_pixel_indices[i]
      lsb = channel_values[value_index] & 1
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
      label2 = "Number of noisy pixels:".ljust(40)
      label3 = "Header size:".ljust(40)
      label4 = "Payload length (from header):".ljust(40)
      
      print()
      print(f"{label1} {width}x{height}")
      print(f"{label2} {num_noisy_pixels:n}")
      print(f"{label3} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label4} {payload_length_bits:n} bits ({payload_length_bits / 8:n} bytes)")
    
    # Extract message bits from noisy pixels
    message_bits = []
    start_index = self.header_len_bits
    end_index = start_index + payload_length_bits
    
    for i in range(start_index, end_index):
      value_index = noisy_pixel_indices[i]
      lsb = channel_values[value_index] & 1
      message_bits.append(lsb)
    
    # Convert bits to bytes (handling partial first byte like iter_bits_from_bytes)
    message_bytes = bytearray()
    
    full_bytes = payload_length_bits // 8
    remaining_bits = payload_length_bits % 8
    
    bit_idx = 0
    
    # First byte may be partial
    if remaining_bits != 0:
      byte_value = 0
      for i in range(remaining_bits):
        byte_value = (byte_value << 1) | message_bits[bit_idx]
        bit_idx += 1
      message_bytes.append(byte_value)
    
    # All remaining bytes are full
    for _ in range(full_bytes):
      byte_value = 0
      for i in range(8):
        byte_value = (byte_value << 1) | message_bits[bit_idx]
        bit_idx += 1
      message_bytes.append(byte_value)
    
    return bytes(message_bytes)


if __name__ == "__main__":
  input_path = Path('./Picture1.png')
  output_image = Path('./output_img.png')
  target_threshold = 0.5
  noice_threshold = 0.1
 
  steg_tool = LsbSobelEdge(noice_threshold)

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
