import secrets
# import string


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


# def generate_random_str_as_bytes(num_storage_bits: int) -> bytes:
#   if num_storage_bits <= 0:
#     return b''
  
#   needed_bytes = num_storage_bits // 8
  
#   if needed_bytes == 0:
#     return b''
  
#   # Set of characters for the randomizer to choose from
#   charset = string.ascii_letters + string.digits + string.punctuation + ' '
    
#   return ''.join(secrets.choice(charset) for _ in range(needed_bytes)).encode('utf-8')


def iter_bits_from_bytes(data: bytes, total_bits: int):
  """
  Iterate over individual bits contained in a byte sequence.

  Bits are yielded in most-significant-bit-first order. If ``total_bits`` is
  not a multiple of 8, the first byte is treated as a partial byte containing
  only ``total_bits % 8`` significant bits. All remaining bytes are processed
  as full 8-bit values.

  Args:
      data: Source byte sequence.
      total_bits: Total number of bits to iterate over.

  Yields:
      int: Individual bit values (0 or 1).

  Returns:
      A generator that produces up to ``total_bits`` bits. Produces no values
      when ``total_bits`` is less than or equal to zero.

  Examples:
      >>> list(iter_bits_from_bytes(b'\\x05', 3))
      [1, 0, 1]

      >>> list(iter_bits_from_bytes(b'\\xA5', 8))
      [1, 0, 1, 0, 0, 1, 0, 1]
  """

  if total_bits <= 0:
    return
    yield  # makes it a generator

  full_bytes = total_bits // 8
  remaining_bits = total_bits % 8

  idx = 0

  # First byte may be partial
  if remaining_bits != 0:
    b = data[0]
    for i in range(remaining_bits - 1, -1, -1):
      yield (b >> i) & 1
    idx = 1

  # All remaining bytes are full
  for j in range(idx, idx + full_bytes):
    b = data[j]
    for i in range(7, -1, -1):
      yield (b >> i) & 1



# if __name__ == "__main__":
#   test_cases = [8, 16, 32, 64, 128, 256]

#   for bits in [9]:
#     message = generate_random_str_as_bytes(bits)
#     storage_bits = len(message) * 8
    
#     print(f"Requested: {bits} bits")
#     print(f"Generated: {len(message)} bytes = {storage_bits} bits")
#     print(f"Match: {bits == storage_bits}")
#     print(f"Sample: {message[:20]}...")
#     print()