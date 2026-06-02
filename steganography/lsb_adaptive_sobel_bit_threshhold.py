
print("Importing modules for LsbSobelEdge...")
from scipy import ndimage
import math
from pathlib import Path
import numpy as np
from PIL import Image
import locale
locale.setlocale(locale.LC_ALL, '')

if __name__ == "__main__":
  from generate import generate_random_bytes
  from ColorModel import ColorModel
else:
  from .generate import generate_random_bytes
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


  def encode_message(self, 
    image_path: Path, 
    output_path: Path,
    target_threshold: float,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    """
    Encode random data into an image using sobel adaptive LSB steganography.
    """

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

     # Find noisy pixels
    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")
    
    noisy_pixel_indices, _ = self._find_n_noisy_pixels(pixel_array, bits_to_encode, verbose)

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
    header_bits = np.unpackbits(np.frombuffer(header_in_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(message_in_bytes, dtype=np.uint8))[-payload_length_bits:]
    all_bits = np.concatenate((header_bits, payload_bits))

    used_indices = noisy_pixel_indices[:len(all_bits)]

    channel_values[used_indices] &= 0xFE
    channel_values[used_indices] |= all_bits

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

    # Flatten the gradient magnitude array
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

    return noisy_pixel_indices, noisy_pixel_indices.size


  def decode_message(self, 
    stego_image_path: Path,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    """
    Decode a message from an image embedded by this class.
    """

    if not Path(stego_image_path).exists() or not Path(stego_image_path).is_file():
      raise FileNotFoundError(f"Stego image file not found: {stego_image_path}")
    
    # Image processing
    img = Image.open(stego_image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.array(img, dtype=np.uint8)
    channel_values = pixel_array.flatten()
    
    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only Grayscale is currently implemented")
    
    # Find noisy pixels for header only
    header_noisy_pixels, _ = self._find_n_noisy_pixels(pixel_array, self.header_len_bits, verbose=False)
    
    # Extract header bits from noisy pixels
    header_bits = (
      channel_values[header_noisy_pixels[:self.header_len_bits]] & 1
    )
    
    # Convert header bits to bytes (MSB-first within each byte)
    header_bytes = np.packbits(header_bits).tobytes()
    
    # Convert bytes to integer (big-endian, matching the encoding)
    payload_length_bits = int.from_bytes(header_bytes, byteorder="big")
    
    # find all noisy pixels needed (header + payload)
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
    
    # Extract message bits from noisy pixels (skip header bits)
    message_bits = (
      channel_values[
        noisy_pixel_indices[self.header_len_bits: total_bits_needed]
      ] & 1
    )
    
    # Handle padded bits
    pad_bits = (-payload_length_bits) % 8

    if pad_bits:
      message_bits = np.concatenate((
        np.zeros(
          pad_bits,
          dtype=np.uint8
        ),
        message_bits
      ))

    return np.packbits(message_bits).tobytes()



if __name__ == "__main__":
  input_path = Path('./cover/Picture1.png')
  output_image = Path('./stego/output_img.png')
  target_threshold = 0.5
 
  steg_tool = LsbSobelEdge()

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
