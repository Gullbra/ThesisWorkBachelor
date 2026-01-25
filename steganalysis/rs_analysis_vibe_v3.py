import numpy as np
from PIL import Image
import argparse
import sys

class RSStegoanalysis:
  def __init__(self, image_path, group_size=4):
    self.image_path = image_path
    self.group_size = group_size
    self.image = Image.open(image_path)
    
    # Ensure we handle RGB/RGBA properly
    if self.image.mode not in ('RGB', 'RGBA', 'L'):
      self.image = self.image.convert('RGB')
      
  def _get_rs_counts(self, data, mask):
    """Vectorized calculation of R and S counts for a specific mask."""
    n_groups = len(data) // self.group_size
    groups = data[:n_groups * self.group_size].reshape(n_groups, self.group_size).astype(np.int16)
    
    # Original discrimination
    f_orig = np.sum(np.abs(np.diff(groups, axis=1)), axis=1)
    
    # Apply Flipping F_1 (LSB Replacement flip)
    # Mask 1: flip, Mask 0: stay, Mask -1: dual flip
    flipped_groups = groups.copy()
    for i in range(self.group_size):
      m = mask[i]
      if m == 1:
        # Standard LSB flip: 0<->1, 2<->3
        flipped_groups[:, i] = groups[:, i] ^ 1
      elif m == -1:
        # Dual LSB flip: -1<->0, 1<->2, 3<->4
        # Implementation: (x+1) XOR 1 - 1
        flipped_groups[:, i] = ((groups[:, i].astype(np.int32) + 1) ^ 1) - 1
    
    f_flipped = np.sum(np.abs(np.diff(flipped_groups, axis=1)), axis=1)
    
    rm = np.sum(f_flipped > f_orig)
    sm = np.sum(f_flipped < f_orig)
    return rm, sm

  def _solve_p(self, d0, d1, dn0, dn1):
    """Solves the RS quadratic equation for message length p."""
    # Coefficients for ax^2 + bx + c = 0
    a = 2 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3 * d0
    c = d0
    
    if a == 0: # Linear fallback
      return abs(c / b) if b != 0 else 0
    
    roots = np.roots([a, b, c])
    
    # We look for the root with the smallest absolute value
    # as it is the most physically plausible for RS
    p = min(roots, key=abs)
    return float(np.real(p))

  def analyze_channel(self, channel_data):
    """Performs RS analysis on a single 1D array of pixel values."""
    # Define standard mask [0,1,1,0] and negative mask [0,-1,-1,0]
    # These are commonly used in literature for group_size=4
    mask = np.array([0, 1, 1, 0])
    neg_mask = -mask
    
    # 1. Counts for M
    rm, sm = self._get_rs_counts(channel_data, mask)
    # 2. Counts for -M
    rm_neg, sm_neg = self._get_rs_counts(channel_data, neg_mask)
    
    # 3. Repeat with 100% flipped LSBs to get d1 and dn1
    flipped_data = channel_data ^ 1
    rm_1, sm_1 = self._get_rs_counts(flipped_data, mask)
    rm_neg_1, sm_neg_1 = self._get_rs_counts(flipped_data, neg_mask)
    
    # Differences
    d0 = rm - sm
    dn0 = rm_neg - sm_neg
    d1 = rm_1 - sm_1
    dn1 = rm_neg_1 - sm_neg_1
    
    p = self._solve_p(d0, d1, dn0, dn1)
    
    # Calculate final ratio
    res = p / (p - 0.5) if (p - 0.5) != 0 else 0
    return max(0.0, min(1.0, abs(res))), (rm, sm, rm_neg, sm_neg)

  def run(self):
    results = {}
    # Handle grayscale vs color
    if self.image.mode == 'L':
      channels = [('Grayscale', np.array(self.image).flatten())]
    else:
      arr = np.array(self.image)
      channels = [
        ('Red', arr[:, :, 0].flatten()),
        ('Green', arr[:, :, 1].flatten()),
        ('Blue', arr[:, :, 2].flatten())
      ]
      
    for name, data in channels:
      ratio, stats = self.analyze_channel(data)
      results[name] = {
        'ratio': ratio,
        'rm': stats[0], 'sm': stats[1],
        'rmn': stats[2], 'smn': stats[3]
      }
    return results

def main():
  parser = argparse.ArgumentParser(description='Advanced RS Stegoanalysis')
  parser.add_argument('path', help='Image path')
  parser.add_argument('--size', type=int, default=4, help='Group size (default: 4)')
  args = parser.parse_args()

  try:
    analyzer = RSStegoanalysis(args.path, args.size)
    results = analyzer.run()
    
    print(f"\n{'Channel':<10} | {'Ratio':<10} | {'Status'}")
    print("-" * 40)
    for channel, data in results.items():
      stego_likely = "DETECTED" if data['ratio'] > 0.05 else "CLEAN"
      print(f"{channel:<10} | {data['ratio']:.2%}   | {stego_likely}")
      
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

if __name__ == '__main__':
  main()