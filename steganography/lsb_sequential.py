print("Importing modules for LsbSequential...")
from pathlib import Path
from PIL import Image
import numpy as np
import locale

locale.setlocale(locale.LC_ALL, '')

if __name__ == "__main__":
    from generate import generate_random_bytes
    from ColorModel import ColorModel
else:
    from .generate import generate_random_bytes
    from .ColorModel import ColorModel

from numpy.lib.stride_tricks import sliding_window_view
print("Modules imported.")


class LsbSequential:
  def __init__(self):
    pass


  no_channels = {
    ColorModel.GRAYSCALE: 1,
    ColorModel.RGB: 3
  }


  def encode_message(
    self,
    image_path: Path,
    target_threshold: float,
    output_path: Path,
    verbose: bool = False,
    color_model: ColorModel = ColorModel.GRAYSCALE,
    delimiter: str = "###END###"
  ) -> bytes:
    """
    Encode random data into an image using LSB sequential steganography.
    """

    if target_threshold < 0 or target_threshold > 1:
      raise ValueError("target_threshold must be between 0 and 1")

    if color_model not in self.no_channels:
      raise ValueError(f"Unsupported image mode: {color_model}. Supported modes: {list(self.no_channels.keys())}")

    if not image_path.exists() or not image_path.is_file():
      raise FileNotFoundError(f"Image file not found: {image_path}")

    if not output_path.parent.exists():
      raise FileNotFoundError("Stego directory does not exist.")

    # Open image as mode and convert to numpy array
    img = Image.open(image_path).convert(color_model)
    img_array = np.array(img)
    width, height = img.size

    # Storage capacity
    capacity_total_bits = img_array.size
    delimiter_bytes = delimiter.encode("ascii")
    delimiter_bits = len(delimiter_bytes) * 8
    capacity_netto_bits = capacity_total_bits - delimiter_bits
    if capacity_netto_bits <= 0:
      raise ValueError("Image too small to encode delimiter.")

    no_of_bits_to_encode = int(target_threshold * capacity_netto_bits)

    if verbose:
      print()
      print(
        f"{'Image dimensions:'.ljust(40)} {width}x{height}"
      )
      print(
        f"{'Delimiter size:'.ljust(40)} "
        f"{delimiter_bits:n} bits "
        f"({delimiter_bits // 8:n} bytes)"
      )
      print(
        f"{'Total capacity:'.ljust(40)} "
        f"{capacity_total_bits:n} bits "
        f"({capacity_total_bits // 8:n} bytes)"
      )
      print(
        f"{'Net capacity:'.ljust(40)} "
        f"{capacity_netto_bits:n} bits "
        f"({capacity_netto_bits // 8:n} bytes)"
      )
      print(
        f"{'Target threshold:'.ljust(40)} "
        f"{target_threshold * 100:.2f}%"
      )
      print(
        f"{'Bits to encode:'.ljust(40)} "
        f"{no_of_bits_to_encode:n} bits "
        f"({no_of_bits_to_encode / 8:.2f} bytes)"
      )

    # Generate bytes to encode
    secret_message = generate_random_bytes(no_of_bits_to_encode)
    full_message = secret_message + delimiter_bytes
    total_message_bits = (no_of_bits_to_encode + delimiter_bits)

    if verbose:
      print(
        f"{'Generated message length:'.ljust(40)} "
        f"{len(secret_message):n} bytes"
      )
      print(
        f"{'Delimiter length:'.ljust(40)} "
        f"{len(delimiter_bytes):n} bytes"
      )

    # Convert to bit stream, and remove leading zeos (from generate_random_bytes if the bits are not divisible by 8) 
    message_bits = np.unpackbits(np.frombuffer(full_message, dtype=np.uint8))[-total_message_bits:]

    # embedd
    flat = img_array.reshape(-1).copy()

    flat[:total_message_bits] &= 0xFE
    flat[:total_message_bits] |= message_bits

    modified_array = flat.reshape(img_array.shape)

    result_img = Image.fromarray(
      modified_array,
      mode=color_model
    )

    result_img.save(output_path, "PNG")

    if verbose:
      print("Random data encoded successfully!")

    print(
      f"Stego image saved to "
      f"{output_path.resolve()}"
    )

    return secret_message


  def decode_message(
    self,
    image_path: Path,
    delimiter: str = "###END###",
    color_model: ColorModel = ColorModel.GRAYSCALE
  ) -> bytes:
    """
    Decode a message from an image using sequential LSB steganography.
    """
    # Open image and convert to numpy array
    img = Image.open(image_path).convert(color_model)
    img_array = np.array(img)

    # Flatten image array
    flat = img_array.reshape(-1)

    # Extract all LSBs
    all_lsbs = flat & 1

    # Convert delimiter to bit list
    delimiter_bits = np.unpackbits(
      np.frombuffer(
        delimiter.encode("ascii"),
        dtype=np.uint8
      )
    )
    delimiter_length = len(delimiter_bits)

    if len(all_lsbs) < delimiter_length:
      raise ValueError(
        "Image too small to contain delimiter."
      )

    # Search for delimiter with sliding window
    windows = sliding_window_view(
      all_lsbs,
      delimiter_length
    )
    indices = np.flatnonzero(
      np.all(
        windows == delimiter_bits,
        axis=1
      )
    )

    if len(indices) == 0:
      raise ValueError("No message found or message corrupted.")

    # Picking first found. A message containing the delimiter is of course not going to work correctly
    found_at = int(indices[0])

    # Extract message bits (before delimiter)
    message_bits = all_lsbs[:found_at]

    if len(message_bits) == 0:
      return b""

    # Reconstruct message
    result = bytearray()
    
    # Handle first potentially incomplete byte
    first_byte_bits = len(message_bits) % 8
    if first_byte_bits:
      partial = message_bits[:first_byte_bits]

      value = 0
      for bit in partial:
        value = (value << 1) | int(bit)

      result.append(value)

      message_bits = message_bits[first_byte_bits:]

    # Handle the rest
    if len(message_bits):
      packed = np.packbits(message_bits)

      result.extend(packed.tobytes())

    return bytes(result)



if __name__ == "__main__":
  steg_tool = LsbSequential()

  input_path = Path('./cover/Picture1.png')
  target_threshold = 0.5
  output_image = Path('./stego/output_img.png')
  delimiter = "###END###"

  print("Testing lsb sequential")
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
