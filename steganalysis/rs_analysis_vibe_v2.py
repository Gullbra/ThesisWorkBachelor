import numpy as np
from PIL import Image
import argparse


class RSStegoanalysis:
  """
  RS (Regular-Singular) Stegoanalysis for detecting LSB steganography
  in grayscale images.
  """
  
  def __init__(self, image_path):
    """Initialize with image path and load the image."""
    self.image_path = image_path
    self.image = Image.open(image_path).convert('L')
    self.pixels = np.array(self.image, dtype=np.uint8)
    
  def _flip_lsb(self, pixels):
    """Flip the least significant bit of pixels."""
    return pixels ^ 1
  
  def _discrimination_function(self, pixels):
    """
    Calculate the discrimination function f(x) which measures smoothness.
    Returns the sum of absolute differences between adjacent pixels.
    """
    diff = np.abs(np.diff(pixels.astype(np.int16)))
    return np.sum(diff)
  
  def _mask_operation(self, pixels, mask):
    """Apply mask operation: flip LSB if mask bit is 1."""
    return np.where(mask, self._flip_lsb(pixels), pixels)
  
  def _analyze_groups(self, group_size=4):
    """
    Divide image into groups and classify as Regular, Singular, or Unusable.
    
    Args:
      group_size: Number of consecutive pixels in each group (typically 3-7).
             Each group is analyzed to see if flipping LSBs increases
             or decreases smoothness.
    
    Returns:
      Tuple of (Rm_pos, Sm_pos, Rm_neg, Sm_neg) - counts of Regular and
      Singular groups for positive and negative masks.
    """
    # Flatten image to 1D array for sequential processing
    pixels_flat = self.pixels.flatten()
    total_pixels = len(pixels_flat)
    
    # Initialize counters
    Rm_pos, Sm_pos, Um_pos = 0, 0, 0  # Positive mask
    Rm_neg, Sm_neg, Um_neg = 0, 0, 0  # Negative mask
    
    # Create masks: alternating pattern for LSB flipping
    # Positive mask: [1, 0, 1, 0, ...] - flip LSB at odd positions
    # Negative mask: [0, 1, 0, 1, ...] - flip LSB at even positions
    mask_pos = np.array([i % 2 for i in range(group_size)])
    mask_neg = 1 - mask_pos
    
    # Process groups of consecutive pixels
    for i in range(0, total_pixels - group_size + 1, group_size):
      # Extract group of consecutive pixels
      group = pixels_flat[i:i+group_size]
      
      if len(group) != group_size:
        continue
      
      # Original discrimination value (smoothness measure)
      f_orig = self._discrimination_function(group)
      
      # Apply positive mask and measure smoothness
      group_pos = self._mask_operation(group, mask_pos)
      f_pos = self._discrimination_function(group_pos)
      
      # Classify group based on smoothness change
      if f_pos > f_orig:
        Rm_pos += 1  # Regular: became less smooth
      elif f_pos < f_orig:
        Sm_pos += 1  # Singular: became more smooth
      else:
        Um_pos += 1  # Unusable: no change
      
      # Apply negative mask and measure smoothness
      group_neg = self._mask_operation(group, mask_neg)
      f_neg = self._discrimination_function(group_neg)
      
      # Classify group based on smoothness change
      if f_neg > f_orig:
        Rm_neg += 1  # Regular: became less smooth
      elif f_neg < f_orig:
        Sm_neg += 1  # Singular: became more smooth
      else:
        Um_neg += 1  # Unusable: no change
    
    return Rm_pos, Sm_pos, Rm_neg, Sm_neg
  
  def estimate_message_length(self):
    """
    Estimate the relative length of hidden message using RS analysis.
    Returns a value between 0 and 1, where 0 means no hidden data
    and 1 means maximum embedding.
    """
    Rm_pos, Sm_pos, Rm_neg, Sm_neg = self._analyze_groups()
    
    # Avoid division by zero
    if Rm_pos + Sm_pos == 0 or Rm_neg + Sm_neg == 0:
      return 0.0
    
    # Calculate d_0 (expected difference for cover image)
    d_pos = Rm_pos - Sm_pos
    d_neg = Rm_neg - Sm_neg
    
    # Solve for p (message length ratio)
    # Using the quadratic approximation: p ≈ (d_pos - d_neg) / (d_pos + d_neg)
    denominator = d_pos + d_neg
    
    if abs(denominator) < 1e-10:
      return 0.0
    
    p = abs(2 * (d_neg / denominator))
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, p))
  
  def analyze(self):
    """
    Perform RS stegoanalysis and return results.
    """
    message_ratio = self.estimate_message_length()
    
    # Get group statistics
    Rm_pos, Sm_pos, Rm_neg, Sm_neg = self._analyze_groups()
    
    # Determine if steganography is likely present
    # Threshold of 0.05 (5%) is commonly used
    threshold = 0.05
    has_stego = message_ratio > threshold
    
    results = {
      'image_path': self.image_path,
      'image_size': self.pixels.shape,
      'estimated_message_ratio': message_ratio,
      'estimated_message_percentage': message_ratio * 100,
      'likely_contains_steganography': has_stego,
      'Rm_positive': Rm_pos,
      'Sm_positive': Sm_pos,
      'Rm_negative': Rm_neg,
      'Sm_negative': Sm_neg,
    }
    
    return results
  
  def print_results(self, results):
    """Print analysis results in a readable format."""
    print("=" * 60)
    print("RS Stegoanalysis Results")
    print("=" * 60)
    print(f"Image: {results['image_path']}")
    print(f"Size: {results['image_size']}")
    print(f"\nEstimated hidden message ratio: {results['estimated_message_ratio']:.4f}")
    print(f"Estimated hidden message percentage: {results['estimated_message_percentage']:.2f}%")
    print(f"\nLikely contains steganography: {results['likely_contains_steganography']}")
    print(f"\nGroup Statistics:")
    print(f"  Rm (positive mask): {results['Rm_positive']}")
    print(f"  Sm (positive mask): {results['Sm_positive']}")
    print(f"  Rm (negative mask): {results['Rm_negative']}")
    print(f"  Sm (negative mask): {results['Sm_negative']}")
    print("=" * 60)


def main():
  """Executive method to run RS stegoanalysis from command line."""
  parser = argparse.ArgumentParser(
    description='Perform RS stegoanalysis on a grayscale image'
  )
  parser.add_argument(
    'image_path',
    type=str,
    help='Path to the grayscale image to analyze'
  )
  
  args = parser.parse_args()
  
  try:
    # Create analyzer instance
    analyzer = RSStegoanalysis(args.image_path)
    
    # Perform analysis
    results = analyzer.analyze()
    
    # Print results
    analyzer.print_results(results)
    
  except FileNotFoundError:
    print(f"Error: Image file '{args.image_path}' not found.")
    return 1
  except Exception as e:
    print(f"Error during analysis: {str(e)}")
    return 1
  
  return 0


if __name__ == '__main__':
  exit(main())
  