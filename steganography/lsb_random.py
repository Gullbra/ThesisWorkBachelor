print("Importing modules for LsbRandom...")
from pathlib import Path
import math
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


class LsbRandom:
  def __init__(self, key):
    if not isinstance(key, int):
      raise ValueError("Key must be an integer (PRNG seed).")
      
    self.key = key
    self.header_len_bits = 32


  bits_per_channel = {
    ColorModel.GRAYSCALE: [1],
    ColorModel.RGB: [3, 3, 2]
  }


  def encode_message(
    self,
    image_path: Path,
    output_path: Path,
    target_threshold: float,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    """
    Encode random data into an image using random LSB steganography.
    """

    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    if color_model not in list(ColorModel):
      raise ValueError(f"Unsupported image mode: {color_model}. Supported modes: {list(ColorModel)}")
    
    if not Path(image_path).exists() or not Path(image_path).is_file():
      raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not output_path.parent.exists():
      raise FileNotFoundError("Stego directory does not exist.")

    # Image Processing
    img = Image.open(image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.asarray(img, dtype=np.uint8)
    channel_values = pixel_array.reshape(-1)

    # Capacity
    embedding_total_capacity = channel_values.size
    bits_to_encode = math.floor(target_threshold * embedding_total_capacity)
    payload_length_bits = (bits_to_encode - self.header_len_bits)

    if payload_length_bits <= 0:
      raise ValueError("Payload length must be positive.")

    if verbose:
      print()
      print(
        f"{'Image dimensions:'.ljust(40)} "
        f"{width}x{height}"
      )
      print(
        f"{'Header size:'.ljust(40)} "
        f"{self.header_len_bits} bits"
      )
      print(
        f"{'Total capacity:'.ljust(40)} "
        f"{embedding_total_capacity:n} bits"
      )
      print(
        f"{'Payload size:'.ljust(40)} "
        f"{payload_length_bits:n} bits"
      )

    # Generate message and embedding path
    message_bytes = generate_random_bytes(payload_length_bits)
    header_bytes = self._generate_header_bytes(payload_length_bits)
    header_bits = np.unpackbits(
      np.frombuffer(
        header_bytes,
        dtype=np.uint8
      )
    )
    payload_bits = np.unpackbits(
      np.frombuffer(
        message_bytes,
        dtype=np.uint8
      )
    )[-payload_length_bits:]
    all_bits = np.concatenate((header_bits, payload_bits))
    embedding_path = self._get_shuffled_indices(channel_values.size)
    used_indices = embedding_path[:len(all_bits)]

    # Embedd
    modified = channel_values.copy()

    modified[used_indices] &= 0xFE
    modified[used_indices] |= all_bits

    # Reprocess pixels and save
    modified_array = modified.reshape(pixel_array.shape)
    Image.fromarray(modified_array,mode=color_model.value).save(output_path)

    return message_bytes



  def decode_message(
    self,
    stego_image_path: Path,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    verbose: bool = False,
  ) -> bytes:
    """
    Decode message encoded with random LSB steganography.
    """

    if color_model not in list(ColorModel):
      raise ValueError(f"Unsupported image mode: {color_model}")

    # Image Processing
    img = Image.open(stego_image_path).convert(color_model)
    width, height = img.size
    pixel_array = np.asarray(img, dtype=np.uint8)
    channel_values = pixel_array.reshape(-1)

    if color_model != ColorModel.GRAYSCALE:
      raise NotImplementedError("Only grayscale currently implemented")

    embedding_path = self._get_shuffled_indices(channel_values.size)

    # Extract header to determine payload length
    header_indices = embedding_path[:self.header_len_bits]
    header_bits = (channel_values[header_indices] & 1)
    header_bytes = np.packbits(header_bits).tobytes()
    payload_length_bits = int.from_bytes(header_bytes, byteorder="big")

    if verbose:
      print()
      print(
        f"{'Image dimensions:'.ljust(40)} "
        f"{width}x{height}"
      )
      print(
        f"{'Header size:'.ljust(40)} "
        f"{self.header_len_bits} bits"
      )
      print(
        f"{'Payload size:'.ljust(40)} "
        f"{payload_length_bits:n} bits"
      )

    # Extract payload bits
    payload_indices = embedding_path[self.header_len_bits: self.header_len_bits + payload_length_bits]
    payload_bits = (channel_values[payload_indices] & 1)
    pad_bits = (-payload_length_bits) % 8

    if pad_bits:
      payload_bits = np.concatenate(
        (
          np.zeros(
              pad_bits,
              dtype=np.uint8
          ),
          payload_bits
        )
      )

    # Convert payload bits to bytes
    message_bytes = np.packbits(payload_bits).tobytes()

    return message_bytes


  def _generate_header_bytes(self, payload_len_bits: int) -> bytes:
    if payload_len_bits < 0 or payload_len_bits >= 2**self.header_len_bits:
      raise ValueError("Payload too large to fit in header.")

    header_len_bytes = self.header_len_bits // 8  # 4 bytes
    return payload_len_bits.to_bytes(header_len_bytes, byteorder="big")


  def _get_shuffled_indices(self, length: int) -> np.ndarray:
    rng = np.random.default_rng(self.key)
    return rng.permutation(length)



if __name__ == "__main__":

  input_path = Path('./cover/Picture1.png')
  output_image = Path('./stego/output_img.png')
  target_threshold = 0.5
  sercret_key = 123456789
 
  steg_tool = LsbRandom(sercret_key)

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

  print(f"\ngenerated msg lenght: {len(generated_message)}\ndecoded msg lenght: {len(decoded_message)}")
  print(f"equal: {generated_message == decoded_message}")
  print(f"First 8 bytes generated: {generated_message[:8]}")
  print(f"First 8 bytes decoded: {decoded_message[:8]}")
