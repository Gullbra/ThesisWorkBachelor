print("Importing modules for RS Steganalysis...")
from enum import Enum
from pathlib import Path
from PIL import Image
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, '')
print("Modules imported.\n")


class ColorModel(Enum):
  RGB = 'RGB'
  HSV = 'HSV'
  YCbCr = 'YCbCr'
  GRAYSCALE = 'L'

colorModelSpecifics = {
  ColorModel.RGB: {
    "channels": 3,
    "channel_names": ["R", "G", "B"]
  },
  ColorModel.HSV: {
    "channels": 3,
    "channel_names": ["H", "S", "V"]
  },
  ColorModel.YCbCr: {
    "channels": 3,
    "channel_names": ["Y", "Cb", "Cr"]
  },
  ColorModel.GRAYSCALE: {
    "channels": 1,
    "channel_names": ["Gray"]
  }
}


class GroupingMethodType(Enum):
  LINEAR = 'linear'
  REAL_ADJACENCY = 'real_adjacency'
  NEIGHBORHOOD = 'neighborhood'

class GroupingMethodWrapper:
  def __init__(self):
    pass

  def group(self, img_array: np.ndarray, color_model: ColorModel, group_size: int, method: GroupingMethodType):
    if method not in GroupingMethodType:
      raise ValueError(f"Unsupported grouping method: {method}")
  
    if method == GroupingMethodType.LINEAR:
      return self._linear_grouping(img_array, color_model, group_size)
    if method == GroupingMethodType.REAL_ADJACENCY:
      return self._real_adjacency_grouping(img_array, group_size)
    if method == GroupingMethodType.NEIGHBORHOOD:
      return self._neighborhood_grouping(img_array)

  def _linear_grouping(self, img_array: np.ndarray, color_model: ColorModel, group_size: int):
    c = colorModelSpecifics[color_model]['channels']
    flat = img_array.reshape(-1, c)
    trimmed = flat[: (flat.shape[0] // group_size) * group_size] # drop remainder pixels if not divisible
    groups = trimmed.reshape(-1, group_size, c)
    return groups

  def _real_adjacency_grouping(self, img_array: np.ndarray, group_size: int):
    k = int(group_size ** 0.5)

    h, w, c = img_array.shape

    img_cropped = img_array[:h - h % k, :w - w % k]

    groups = img_cropped.reshape(h//k, k, w//k, k, c)
    groups = groups.transpose(0, 2, 1, 3, 4)
    groups = groups.reshape(-1, k*k, c)
    return groups
  
  def _neighborhood_grouping(self, img_array: np.ndarray):
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(img_array, (2, 2, img_array.shape[2]))
    groups = windows.reshape(-1, 4, img_array.shape[2])
    return groups


class DiscriminationFunctionType(Enum):
  ABS_DIFF_SUM = 'abs_diff_sum'

class DiscriminationFunctionWrapper:
  def __init__(self):
    pass

  def calculate(self, pixel_values: np.ndarray, method: DiscriminationFunctionType):
    if method not in DiscriminationFunctionType:
      raise ValueError(f"Unsupported discrimination function method: {method}")
    
    if method == DiscriminationFunctionType.ABS_DIFF_SUM:
      return self._abs_diff_sum(pixel_values)

  def _abs_diff_sum(self, pixel_values: np.ndarray):
    return np.sum(np.abs(np.diff(pixel_values))) 


def F1(x):
  if x % 2 == 0:
    return min(x + 1, 255)
  else:
    return max(x - 1, 0)

def Fm1(x):
  if x % 2 == 0:
    return max(x - 1, 0)
  else:
    return min(x + 1, 255)

flipping_functions = {
  1: F1,
  -1: Fm1,
  0: lambda x: x
}

# Corrected flipping functions per Fridrich et al. paper
# flipping_functions = {
#   1: lambda x: x ^ 1,  # F1: flip LSB unconditionally
#   -1: lambda x: x ^ 1 if x % 2 == 1 else x,  # F-1: flip LSB only if pixel is odd
#   0: lambda x: x  # F0: no change
# }


class RSAnalysis:
  def __init__(self):
    self._GroupingMethods = GroupingMethodWrapper()
    self._DiscriminationFunctions = DiscriminationFunctionWrapper()

  def _generate_negative_mask(self, mask_p):
    """Generate -M mask by swapping 1s and -1s in M"""
    return [(-1 if m == 1 else (1 if m == -1 else 0)) for m in mask_p]

  def _estimate_m_length(self, original_stats, inverted_stats):
    """
    Estimate message length using the characteristic function approach.
    Based on Fridrich et al. equation for estimating p (embedding rate).
    """
    results = {}
    for ch in original_stats.keys():
      # Get counts from original image
      R_M = original_stats[ch]['M']['R']
      S_M = original_stats[ch]['M']['S']
      R_nM = original_stats[ch]['-M']['R']
      S_nM = original_stats[ch]['-M']['S']
      
      # Get counts from inverted image
      R_M_inv = inverted_stats[ch]['M']['R']
      S_M_inv = inverted_stats[ch]['M']['S']
      R_nM_inv = inverted_stats[ch]['-M']['R']
      S_nM_inv = inverted_stats[ch]['-M']['S']
      
      # Calculate total groups for normalization
      total_M = R_M + S_M + original_stats[ch]['M']['U']
      total_nM = R_nM + S_nM + original_stats[ch]['-M']['U']
      
      # Normalize to get proportions
      r_M = R_M / total_M if total_M > 0 else 0
      s_M = S_M / total_M if total_M > 0 else 0

      r_nM = R_nM / total_nM if total_nM > 0 else 0
      s_nM = S_nM / total_nM if total_nM > 0 else 0

      r_M_inv = R_M_inv / total_M if total_M > 0 else 0
      s_M_inv = S_M_inv / total_M if total_M > 0 else 0

      # Calculate d values (discriminator differences)
      d_0 = r_M - s_M
      d_m0 = r_nM - s_nM
      d_1 = r_M_inv - s_M_inv
      
      # Solve quadratic equation: 2(d1+d0)x^2 + (dm0-d1-3d0)x + d0-dm0 = 0
      # This gives x = proportion of pixels with embedded message
      a = 2 * (d_1 + d_0)
      b = d_m0 - d_1 - 3 * d_0
      c = d_0 - d_m0
      
      if abs(a) < 1e-10:  # Effectively zero
        if abs(b) < 1e-10:
          x = 0
        else:
          x = -c / b
      else:
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
          x = 0  # No real solution
        else:
          # Two roots - choose the one in [0, 1] range
          x1 = (-b + np.sqrt(discriminant)) / (2*a)
          x2 = (-b - np.sqrt(discriminant)) / (2*a)
          
          # Pick the root closest to the valid range [0, 1]
          candidates = []
          if 0 <= x1 <= 1:
            candidates.append(x1)
          if 0 <= x2 <= 1:
            candidates.append(x2)
          
          if candidates:
            x = min(candidates, key=lambda val: abs(val - 0.5))
          else:
            # If no root in valid range, pick the closest one
            x = min([x1, x2], key=lambda val: min(abs(val), abs(val - 1)))
      
      # # Convert x to embedding rate p
      # # Relationship: x ≈ p/2 for small p, or p ≈ x/(x - 0.5) for general case
      # if abs(x - 0.5) < 1e-10:
      #   p = 0
      # else:
      #   p = abs(x) / abs(x - 0.5) if x != 0 else 0
      
      # # Clamp to reasonable range [0, 1]
      # p = max(0, min(1, p))
      
      p = max(0, min(1, 2 * x))
      
      results[ch] = {
        'percentage': f"{p * 100:.2f}%",
        'value': p,
        'x': x
      }
      
    return results

  def _analyze_image(
    self,
    img_array: np.ndarray, 
    color_model: ColorModel,
    group_size: int,
    mask_p: list[int],
    mask_n: list[int],
    grouping_method: GroupingMethodType,
    discrimination_function: DiscriminationFunctionType,
    verbose: bool
  ):
    groups = self._GroupingMethods.group(img_array, color_model, group_size, grouping_method)

    if verbose:
      print(f"Total groups formed: {groups.shape[0]:n} using {grouping_method.value} method.")
      print()

    state_dict = {}
    for channel_key in colorModelSpecifics[color_model]['channel_names']:
      state_dict[channel_key] = {
        "M": {
          "mask": mask_p,
          "R": 0,
          "S": 0,
          "U": 0,
        },
        "-M": {
          "mask": mask_n,
          "R": 0,
          "S": 0,
          "U": 0
        }
      }

    for p_group in groups:
      for channel_index in range(colorModelSpecifics[color_model]['channels']):
        channel_key = colorModelSpecifics[color_model]['channel_names'][channel_index]
        for mask_key in state_dict[channel_key]:

          pixel_values_for_channel = p_group[:, channel_index].astype(np.int32)

          flipped_pixels = np.array([
            flipping_functions[m](p) for p, m in zip(pixel_values_for_channel, state_dict[channel_key][mask_key]['mask'])
          ])

          discrim_unflipped = self._DiscriminationFunctions.calculate(
            pixel_values_for_channel, 
            discrimination_function
          )
          discrim_flipped = self._DiscriminationFunctions.calculate(
            flipped_pixels,
            discrimination_function
          )

          if (discrim_flipped > discrim_unflipped):
            state_dict[channel_key][mask_key]['R'] += 1
          elif (discrim_flipped < discrim_unflipped):
            state_dict[channel_key][mask_key]['S'] += 1
          else:
            state_dict[channel_key][mask_key]['U'] += 1
    
    if verbose:
      print("RS Analysis Results:")
      for channel_key, res_by_channel in state_dict.items():
        print(f"Channel: {channel_key}")
        for mask_key in res_by_channel:
          R = res_by_channel[mask_key]['R']
          S = res_by_channel[mask_key]['S']
          U = res_by_channel[mask_key]['U']
          print(f"  Mask {mask_key}: R = {R:n}, S = {S:n}, U = {U:n}")
      print()

    return state_dict
  
  def analyze_image(
    self,
    path_to_image: Path, 
    color_model: ColorModel = ColorModel.RGB,
    group_size: int = 4,
    mask_p = [1, 0, -1, 0],
    grouping_method: GroupingMethodType = GroupingMethodType.LINEAR,
    discrimination_function: DiscriminationFunctionType = DiscriminationFunctionType.ABS_DIFF_SUM,
    verbose: bool = False
  ):
    if not Path(path_to_image).is_file():
      raise FileNotFoundError(f"Image file not found: {path_to_image}")
    
    if color_model not in colorModelSpecifics:
      raise ValueError(f"Unsupported color model: {color_model}")
    
    if len(mask_p) != group_size:
      raise ValueError(f"Mask length {len(mask_p)} does not match group size {group_size}.")

    img = Image.open(path_to_image).convert(color_model.value)
    img_array = np.array(img)

    if verbose:
      width, height = img.size
      print(f"Image opened: {path_to_image}")
      print(f"Dimensions: {width}x{height}")
      print(f"Color Model: {color_model.value} ({colorModelSpecifics[color_model]['channels']} channels)")
      print(f"Group Size: {group_size}")
      print(f"Mask M: {mask_p}")
      print(f"Mask -M: {self._generate_negative_mask(mask_p)}")
      print()
    
    mask_n = self._generate_negative_mask(mask_p)
    
    res_normal = self._analyze_image(
      img_array,
      color_model,
      group_size,
      mask_p,
      mask_n,
      grouping_method,
      discrimination_function,
      verbose
    )

    inverted_img = np.vectorize(flipping_functions[1])(img_array)

    res_inverted = self._analyze_image(
      inverted_img,
      color_model,
      group_size,
      mask_p,
      mask_n,
      grouping_method,
      discrimination_function,
      verbose
    )

    m_length_estimates = self._estimate_m_length(res_normal, res_inverted)
    if verbose:
      print("Estimated Message Lengths per Channel:")
      for ch, est in m_length_estimates.items():
        print(f"  Channel {ch}: {est['percentage']} (x={est['x']:.4f})")
      print()
    
    return m_length_estimates


if __name__ == "__main__":
  test_analyser = RSAnalysis()

  # Test on cover image (should be close to 0%)
  # test_analyser.analyze_image(
  #   path_to_image = (Path(__file__).parent / "1_cover.png").resolve(),
  #   color_model=ColorModel.GRAYSCALE,
  #   group_size=4,
  #   verbose=True
  # )    

  # Test on stego image (should show embedding percentage)
  test_analyser.analyze_image(
    path_to_image = (Path(__file__).parent / "1_stego.png").resolve(),
    color_model=ColorModel.GRAYSCALE,
    group_size=4,
    verbose=True
  )