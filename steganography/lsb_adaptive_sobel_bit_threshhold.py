
print("Importing modules for LsbSobelEdge test...")
import itertools
import time
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

  def __init__(self):    
    self.header_len_bits = 32  # 32 bits to store the message length in bytes


  no_channels = {
    ColorModel.GRAYSCALE: 1, 
    ColorModel.RGB: 3
  }

  def test_time(self, input_path: Path, output_path: Path, target_threshhold):
    start1 = time.time()
    self.new_encode_message(image_path=input_path, output_path=output_path, target_threshold=target_threshhold)
    end1 = time.time()
    start2 = time.time()
    self.encode_message(image_path=input_path, output_path=output_path, target_threshold=target_threshhold)
    end2 = time.time()

    # print(f"\nnew: {end1 - start1}\nold: {end2-start2}")
    return (end1 - start1, end2-start2)


  def new_encode_message(self, 
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

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")

    # Capacity calculation
    embedding_total_capacity = width * height
    bits_to_encode = math.floor(target_threshold * embedding_total_capacity)
    payload_length_bits = bits_to_encode - self.header_len_bits

    # OPTIMIZATION 1: Inline gradient calculation instead of calling _find_n_noisy_pixels
    img_data = (pixel_array & 0xFE).astype(float)
    gradient_magnitude = np.hypot(
      ndimage.sobel(img_data, axis=1), 
      ndimage.sobel(img_data, axis=0)
    )
    flat_gradient = gradient_magnitude.flatten()
    
    # Find the N noisiest pixels
    num_pixels_needed = min(bits_to_encode, flat_gradient.size)
    partition_indices = np.argpartition(-flat_gradient, num_pixels_needed - 1)
    noisy_pixel_indices = partition_indices[:num_pixels_needed]
    
    # Sort by gradient magnitude (descending order)
    gradient_values = flat_gradient[noisy_pixel_indices]
    sorted_order = np.argsort(-gradient_values)
    noisy_pixel_indices = noisy_pixel_indices[sorted_order]

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
      print(f"First 8 noisy pixel indices:            {noisy_pixel_indices[:8]}")
      print(f"Last 8 noisy pixel indices:             {noisy_pixel_indices[-8:]}")
      print(f"Gradient range of selected pixels:       {flat_gradient[noisy_pixel_indices].min():.2f} to {flat_gradient[noisy_pixel_indices].max():.2f}")

    if bits_to_encode > embedding_total_capacity:
      raise ValueError("Not enough noisy pixels to encode message")
    
    # Generate message
    message_in_bytes = generate_random_bytes(payload_length_bits)
    header_in_bytes = self._generate_header_bytes(payload_length_bits)

    # OPTIMIZATION 2: Pre-allocate and vectorize bit generation
    # Create combined bit array
    total_bits = []
    
    # Add header bits
    full_bytes = self.header_len_bits // 8
    remaining_bits = self.header_len_bits % 8
    idx = 0
    
    if remaining_bits != 0:
      b = header_in_bytes[0]
      for i in range(remaining_bits - 1, -1, -1):
        total_bits.append((b >> i) & 1)
      idx = 1
    
    for j in range(idx, idx + full_bytes):
      b = header_in_bytes[j]
      for i in range(7, -1, -1):
        total_bits.append((b >> i) & 1)
    
    # Add message bits
    full_bytes = payload_length_bits // 8
    remaining_bits = payload_length_bits % 8
    idx = 0
    
    if remaining_bits != 0:
      b = message_in_bytes[0]
      for i in range(remaining_bits - 1, -1, -1):
        total_bits.append((b >> i) & 1)
      idx = 1
    
    for j in range(idx, idx + full_bytes):
      b = message_in_bytes[j]
      for i in range(7, -1, -1):
        total_bits.append((b >> i) & 1)
    
    # OPTIMIZATION 3: Vectorized embedding
    # Convert bits to numpy array for vectorized operations
    bits_array = np.array(total_bits, dtype=np.uint8)
    
    # Clear LSBs and set new bits in one vectorized operation
    channel_values[noisy_pixel_indices[:len(bits_array)]] = \
      (channel_values[noisy_pixel_indices[:len(bits_array)]] & 0xFE) | bits_array

    # Reprocess pixels and save
    pixel_array = channel_values.reshape(height, width, self.no_channels[color_model])
    if self.no_channels[color_model] == 1:
      pixel_array = pixel_array.squeeze()
    Image.fromarray(pixel_array, mode=color_model.value).save(output_path)

    # Returning message for testing
    return message_in_bytes


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

    # Capacity
    embedding_total_capacity = width * height
    bits_to_encode = math.floor(target_threshold * embedding_total_capacity)
    payload_length_bits = bits_to_encode - self.header_len_bits

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")
    
    # Find noisy pixels
    noisy_pixel_indices = self._find_n_noisy_pixels(pixel_array, bits_to_encode, verbose)

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


  def _find_n_noisy_pixels(self, pixel_array: np.ndarray, num_of_pixels: int, verbose: bool) -> np.ndarray:
    """
    Finds the N noisiest pixels in the image based on Sobel gradient magnitude.
    
    Args:
        pixel_array: The image pixel array
        num_of_pixels: Number of noisiest pixels to select
        verbose: Whether to print debug information
    
    Returns:
        Tuple of (noisy_pixel_indices, actual_number_selected)
    """
    # Excluding the least significant bit
    img_data = (pixel_array & 0xFE).astype(float)

    # Sobel matrix operations
    gradient_magnitude = np.hypot(
      ndimage.sobel(img_data, axis=1), 
      ndimage.sobel(img_data, axis=0)
    )

    # Excluding the least significant bit, so that we can decode and encode with the same noise finding method
    flat_gradient = gradient_magnitude.flatten()
    
    # Ensure we don't request more pixels than available
    num_of_pixels = min(num_of_pixels, flat_gradient.size)
    
    # Find indices of the N largest gradient values
    # np.argpartition is more efficient than full sort for top-N selection
    # Using negative values because we want the largest (not smallest)
    partition_indices = np.argpartition(-flat_gradient, num_of_pixels - 1)
    
    # Get the top N indices
    noisy_pixel_indices = partition_indices[:num_of_pixels]
    
    # Sort these indices by their gradient magnitude (descending order)
    # This ensures consistent ordering if the same N is requested
    gradient_values = flat_gradient[noisy_pixel_indices]
    sorted_order = np.argsort(-gradient_values)
    noisy_pixel_indices = noisy_pixel_indices[sorted_order]

    if verbose:
      print(f"Requested pixels:                        {num_of_pixels:n}")
      print(f"Actually selected:                       {noisy_pixel_indices.size:n}")
      print(f"First 10 noisy pixel indices:            {noisy_pixel_indices[:10]}")
      print(f"Last 10 noisy pixel indices:             {noisy_pixel_indices[-10:]}")
      print(f"Gradient range of selected pixels:       {flat_gradient[noisy_pixel_indices].min():.2f} to {flat_gradient[noisy_pixel_indices].max():.2f}")

    return noisy_pixel_indices


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
    
    # Step 1: Find noisy pixels for header only
    header_noisy_pixels, _ = self._find_n_noisy_pixels(pixel_array, self.header_len_bits, verbose=False)
    
    # Step 2: Extract header bits from noisy pixels
    header_bits = []
    for i in range(self.header_len_bits):
      value_index = header_noisy_pixels[i]
      lsb = channel_values[value_index] & 1
      header_bits.append(lsb)
    
    # Step 3: Convert header bits to bytes (MSB-first within each byte)
    header_bytes = bytearray()
    for i in range(0, self.header_len_bits, 8):
      byte_bits = header_bits[i:i+8]
      byte_value = 0
      for bit in byte_bits:
        byte_value = (byte_value << 1) | bit  # MSB first
      header_bytes.append(byte_value)
    
    # Step 4: Convert bytes to integer (big-endian, matching the encoding)
    payload_length_bits = int.from_bytes(header_bytes, byteorder="big")
    
    # Step 5: Now find all noisy pixels needed (header + payload)
    total_bits_needed = self.header_len_bits + payload_length_bits
    noisy_pixel_indices, num_noisy_pixels = self._find_n_noisy_pixels(pixel_array, total_bits_needed, verbose)
    
    if verbose:
      label1 = "Image dimensions:".ljust(40)
      label2 = "Number of noisy pixels needed:".ljust(40)
      label3 = "Header size:".ljust(40)
      label4 = "Payload length (from header):".ljust(40)
      label5 = "Total bits to decode:".ljust(40)
      
      print()
      print(f"{label1} {width}x{height}")
      print(f"{label2} {num_noisy_pixels:n}")
      print(f"{label3} {self.header_len_bits:n} bits ({self.header_len_bits / 8:n} bytes)")
      print(f"{label4} {payload_length_bits:n} bits ({payload_length_bits / 8:n} bytes)")
      print(f"{label5} {total_bits_needed:n} bits ({total_bits_needed / 8:n} bytes)")
    
    # Step 6: Extract message bits from noisy pixels (skip header bits)
    message_bits = []
    start_index = self.header_len_bits
    end_index = start_index + payload_length_bits
    
    for i in range(start_index, end_index):
      value_index = noisy_pixel_indices[i]
      lsb = channel_values[value_index] & 1
      message_bits.append(lsb)
    
    # Step 7: Convert bits to bytes (handling partial first byte like iter_bits_from_bytes)
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
 
  steg_tool = LsbSobelEdge()

  # print("\n...encoding random data")
  # generated_message = steg_tool.encode_message(
  #   image_path=input_path, 
  #   output_path=output_image, 
  #   target_threshold= target_threshold, 
  #   verbose=True
  # )

  # print("\n...decoding image")
  # decoded_message = steg_tool.decode_message(
  #   stego_image_path=output_image,
  #   verbose=True
  # )

  # print(f"\ngenerated msg lenght: {len(generated_message)}\ndecoded msg lenght: {len(decoded_message)}")
  # print(f"equal: {generated_message == decoded_message}")
  # print(f"First 8 bytes generated: {generated_message[:8]}")
  # print(f"First 8 bytes decoded: {decoded_message[:8]}")
  # print(f"Last 8 bytes generated: {generated_message[-8:]}")
  # print(f"Last 8 bytes decoded: {decoded_message[-8:]}")

  results = []
  for i in range(20):
    results.append(
      steg_tool.test_time(
        input_path=input_path,
        output_path=output_image, 
        target_threshhold=target_threshold)
    )

  # print(results)
  index_0_values, index_1_values = zip(*results)

  # Calculate averages
  avg_0 = sum(index_0_values) / len(index_0_values)
  avg_1 = sum(index_1_values) / len(index_1_values)

  print(f"Average new: {avg_0}, Average old: {avg_1}")

