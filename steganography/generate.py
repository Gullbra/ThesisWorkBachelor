import secrets


def generate_random_bytes(num_bits: int) -> bytes:
  """
  Generate a cryptographically secure random byte sequence containing up to
  the specified number of bits.

  The returned byte string is large enough to store ``num_bits`` bits. When
  ``num_bits`` is not a multiple of 8, any unused high-order bits in the
  first byte are cleared to ensure that only the requested number of bits
  contain random data.

  Args:
      num_bits: Number of random bits to generate.

  Returns:
      A byte string containing random data. Returns ``b''`` when
      ``num_bits`` is less than or equal to zero.

  Examples:
      >>> len(generate_random_bytes(16))
      2

      >>> len(generate_random_bytes(9))
      2
  """

  if num_bits <= 0:
    return b''
  
  random_bytes = secrets.token_bytes((num_bits + 7) // 8)
  
  # If num_bits isn't a multiple of 8, we do a bitwise AND on 
  #   * the most significant byte 
  #   * and a mask generated based on the number of remaining bits.
  if num_bits % 8 != 0:
    random_bytes = bytes([random_bytes[0] & ((1 << num_bits % 8) - 1)]) + random_bytes[1:]
  
  return random_bytes
