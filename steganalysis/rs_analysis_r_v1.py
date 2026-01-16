
print("Importing moludles for RS Steganalysis...")
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
  REAL_ADJECENCY = 'real_adjecency'
  NEIGHBORHOOD = 'neighborhood'

class GroupingMethodWrapper:
  def __init__(self):
    pass

  def group(self, img_array: np.ndarray, color_model: ColorModel, group_size: int, method: GroupingMethodType):
    if method not in GroupingMethodType:
      raise ValueError(f"Unsupported grouping method: {method}")
  
    if method == GroupingMethodType.LINEAR:
      return self._linear_grouping(img_array, color_model, group_size)
    if method == GroupingMethodType.REAL_ADJECENCY:
      return self._real_adjecency_grouping(img_array, group_size)
    if method == GroupingMethodType.NEIGHBORHOOD:
      return self._neighborhood_grouping(img_array)

  def _linear_grouping(self, img_array: np.ndarray, color_model: ColorModel, group_size: int):
    c = colorModelSpecifics[color_model]['channels']
    flat = img_array.reshape(-1, c)
    trimmed = flat[: (flat.shape[0] // group_size) * group_size] # drop remainder pixels if not divisible
    groups = trimmed.reshape(-1, group_size, c)

    # print("\ngroups:")
    # print(groups)

    # one_group = groups[0]
    # print("\none group:")
    # print(one_group)

    # next_lvl = one_group[0]
    # print("\next lvl:")
    # print(next_lvl)

    return groups

  def _real_adjecency_grouping(self, img_array: np.ndarray, group_size: int):
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


class DescriminationFunctionType(Enum):
  ABS_DIFF_SUM = 'abs_diff_sum'

class DescriminationFunctionWrapper:
  def __init__(self):
    pass

  def calculate(self, pixel_values: np.ndarray, method: DescriminationFunctionType, channel: int = 0):
    if method not in DescriminationFunctionType:
      raise ValueError(f"Unsupported descrimination function method: {method}")
    
    if method == DescriminationFunctionType.ABS_DIFF_SUM:
      return self._abs_diff_sum(pixel_values, channel)

  def _abs_diff_sum(self, pixel_values: np.ndarray, channel: int = 0):
    diffs = []
    for i in range(len(pixel_values) - 1):
      diff = abs(int(pixel_values[i][channel]) - int(pixel_values[i + 1][channel]))
      diffs.append(diff)
    f_values = np.sum(diffs)
    return f_values


flipping_functions = {
  1: lambda x: x ^ 1,
  -1: lambda x: ((x + 1) ^ 1) - 1,
  0: lambda x: x
}


class RSAnalysis:
  def __init__(self):
    self._GroupingMethods=GroupingMethodWrapper()
    self._DescriminationFunctions=DescriminationFunctionWrapper()

  def analyze_image(
    self,
    path_to_image: Path, 
    color_model: ColorModel = ColorModel.RGB,
    group_size: int = 4,
    mask_p = [1, 0, -1, 0],
    grouping_method: GroupingMethodType = GroupingMethodType.LINEAR,
    descrimination_function: DescriminationFunctionType = DescriminationFunctionType.ABS_DIFF_SUM,
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
      print(f"Color Model: {color_model} ({colorModelSpecifics[color_model]['channels']} channels)")
      print(f"Group Size: {group_size}")
      print()

    groups = self._GroupingMethods.group(img_array, color_model, group_size, grouping_method)

    if verbose:
      print(f"Total groups formed: {groups.shape[0]:n} using {grouping_method} method.")
      print()

    # for channel in range(colorModelSpecifics[color_model]['channels']):

    state_dict = {
      "positive_mask": {
        "mask": mask_p,
        "R": 0,
        "S": 0,
      },
      "negative_mask": {
        "mask": [el * -1 for el in mask_p],
        "R": 0,
        "S": 0,
      }
    }

    for p_group in groups:
      p_group: np.ndarray[tuple[int]]

      for mask_key in state_dict:
        flipped_pixels = np.array([
          flipping_functions[m](p) for p, m in zip(p_group, state_dict[mask_key]['mask'])
        ])

        discrim_unflipped = self._DescriminationFunctions.calculate(
          p_group, 
          descrimination_function
        )
        discrim_flipped = self._DescriminationFunctions.calculate(
          flipped_pixels,
          descrimination_function
        )

        if (discrim_flipped > discrim_unflipped):
          state_dict[mask_key]['R'] += 1
        elif (discrim_flipped < discrim_unflipped):
          state_dict[mask_key]['S'] += 1
    
    print("RS Analysis Results:")
    for mask_key in state_dict:
      R = state_dict[mask_key]['R']
      S = state_dict[mask_key]['S']
      print(f"Mask {state_dict[mask_key]['mask']}: R = {R:n}, S = {S:n}")


if __name__ == "__main__":
  test_analyser = RSAnalysis()

  test_analyser.analyze_image(
    path_to_image = (Path(__file__).parent / "1_cover.png").resolve(),
    color_model=ColorModel.GRAYSCALE,
    group_size=4,
    verbose=True
  )    

  test_analyser.analyze_image(
    path_to_image = (Path(__file__).parent / "1_stego.png").resolve(),
    color_model=ColorModel.GRAYSCALE,
    group_size=4,
    verbose=True
  )    
