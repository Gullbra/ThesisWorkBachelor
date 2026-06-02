print("Importing modules for LsbMatching...")
import math
from pathlib import Path
import numpy as np
from PIL import Image

if __name__ == "__main__":
  from generate import generate_random_bytes
  from ColorModel import ColorModel
else:
  from .generate import generate_random_bytes
  from .ColorModel import ColorModel
print("Modules imported.")


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

    # Generate message
    message_bytes = generate_random_bytes(payload_length_bits)
    header_bytes = self._generate_header_bytes(payload_length_bits)
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))[-payload_length_bits:]
    all_bits = np.concatenate((header_bits, payload_bits))

    # Embedd
    modified = channel_values.copy()

    # relevant_values contains references to the ones in maodified, and modified will be udated along with relevant_values
    relevant_values = modified[:len(all_bits)]

    # find everywhere where the lsb doesn't already match
    mismatch_mask = (relevant_values & 1 != all_bits)

    # the matching
    rng = np.random.default_rng()
    delta = rng.choice(
      np.array([-1, 1], dtype=np.int16),
      size=mismatch_mask.sum()
    )
    values = (relevant_values[mismatch_mask].astype(np.int16)) + delta

    # handle overflow/underflow
    values[values > 255] = 254
    values[values < 0] = 1

    relevant_values[mismatch_mask] = (values.astype(np.uint8))

    Image.fromarray(modified.reshape(pixel_array.shape), mode=color_model.value).save(output_path)

    return message_bytes


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
    header_bits = (channel_values[:self.header_len_bits] & 1)
    header_bytes = np.packbits(header_bits).tobytes()
    
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
    payload_bits = (
      channel_values[self.header_len_bits: self.header_len_bits + payload_length_bits] & 1
    )
    num_pad_bits = (-payload_length_bits) % 8

    if num_pad_bits:
      payload_bits = np.concatenate((
        np.zeros(
          num_pad_bits,
          dtype=np.uint8
        ),
        payload_bits
      ))

    return np.packbits(payload_bits).tobytes()
  


if __name__ == "__main__":
  input_path = Path('./cover/Picture1.png')
  output_image = Path('./stego/output_img.png')
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
