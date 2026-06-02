from enum import Enum


class ColorModel(str, Enum):
  RGB = 'RGB'
  GRAYSCALE = 'L'