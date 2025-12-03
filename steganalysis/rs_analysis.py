import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

class RSAnalysis:
    """
    Implements the core mechanism of RS-Analysis to calculate the percentage 
    of Regular (R) and Singular (S) groups for a given image and mask.
    
    This is the foundational step for determining if LSB steganography has been 
    used based on the statistical distribution of R and S groups.
    """

    def __init__(self, mask=(1, -1)):
        """
        Initializes the RS Analysis with a standard mask.
        :param mask: The discrimination mask M, typically (1, -1) for k=2 groups.
        """
        if len(mask) != 2:
            raise ValueError("The mask must have length 2 for this k=2 implementation.")
        self.mask = np.array(mask)
        self.group_size = 2 # k=2

    def _get_pixel_groups(self, channel_array):
        """
        Divides a 1D pixel channel into non-overlapping groups of size k=2.
        """
        # Ensure the array length is divisible by the group size
        num_groups = channel_array.size // self.group_size
        
        # Reshape the array into groups of size 2
        groups = channel_array[:num_groups * self.group_size].reshape(num_groups, self.group_size)
        return groups

    def _discrimination_function(self, group):
        """
        The discrimination function f(x1, x2) = |x2 - x1|.
        This measures the smoothness of the group.
        """
        return np.abs(group[1] - group[0])

    def _flip_lsb(self, group):
        """
        Applies LSB flipping to the group based on the mask M.
        
        If mask[i] = 1, flip the LSB of group[i].
        If mask[i] = -1, do not flip the LSB of group[i].
        """
        # Create a copy of the group
        flipped_group = np.copy(group)
        
        # Logic: If mask element is 1 (or -1 in the case of -1), flip the LSB.
        # We only flip where mask value is 1 (standard approach)
        
        # The flip operation is: value XOR 1 (if mask is 1)
        # Using a fixed mask of (1, -1) for simplicity here, where only the first LSB is flipped.
        # However, the proper definition applies the mask:
        
        # Apply LSB flipping where mask element is 1 (for R curve calculation)
        for i in range(self.group_size):
            if self.mask[i] == 1:
                # Flip LSB (equivalent to: if LSB is 0 -> 1, if LSB is 1 -> 0)
                flipped_group[i] = flipped_group[i] ^ 1
            # If mask[i] is -1, no flip occurs on that pixel's LSB
            
        return flipped_group

    def _calculate_rs(self, groups, mask):
        """
        Calculates the percentage of Regular (R) and Singular (S) groups.
        
        :param groups: The numpy array of pixel groups.
        :param mask: The mask M used for discrimination.
        :return: Tuple of (R_count, S_count)
        """
        R_count = 0
        S_count = 0
        
        # Set the instance mask temporarily to the provided mask
        self.mask = mask 

        for group in groups:
            # 1. Calculate smoothness of the original group
            f_original = self._discrimination_function(group)
            
            # 2. Apply LSB flipping (where mask is 1)
            f_flipped = self._discrimination_function(self._flip_lsb(group))

            # 3. Classification
            if f_flipped < f_original:
                # Smoothness increased (f decreased) -> Regular
                R_count += 1
            elif f_flipped > f_original:
                # Smoothness decreased (f increased) -> Singular
                S_count += 1
            # If f_flipped == f_original, the group is Unchanged (U) and ignored.

        total_groups = len(groups)
        R_percentage = (R_count / total_groups) * 100
        S_percentage = (S_count / total_groups) * 100
        
        return R_percentage, S_percentage

    def analyze_image(self, image_path):
        """
        Performs the core RS analysis on the image.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img, dtype=np.int16) # Use int16 for safe arithmetic

            # Focus on the Red channel (index 0) for demonstration
            R_channel = img_array[:, :, 0].flatten()
            groups = self._get_pixel_groups(R_channel)
            
            # --- Calculation for R curve ---
            # Mask M = (1, -1). This is used to calculate R and S points.
            mask_M = np.array([1, -1])
            R_M, S_M = self._calculate_rs(groups, mask_M)

            # --- Calculation for R- curve ---
            # Mask -M = (-1, 1). This is used to calculate R- and S- points.
            mask_minus_M = np.array([-1, 1])
            R_minus_M, S_minus_M = self._calculate_rs(groups, mask_minus_M)
            
            # Total groups for percentage calculation
            total_groups = len(groups)
            
            # NOTE: A full RS-Analysis requires plotting these values against 
            # different shift values (rho) to find the intersection point, but 
            # this demonstrates the core calculation for the 0% shift point.
            
            print(f"--- RS Analysis Results for {os.path.basename(image_path)} (Red Channel) ---")
            print(f"Total analyzed groups (k={self.group_size}): {total_groups}")
            print(f"Mask M = {mask_M} (R_M, S_M): R = {R_M:.2f}%, S = {S_M:.2f}%")
            print(f"Mask -M = {mask_minus_M} (R_(-M), S_(-M)): R = {R_minus_M:.2f}%, S = {S_minus_M:.2f}%")
            
            # Simple Detection Logic (0% shift):
            # If R_M > S_M, the image is likely a natural cover or has very low embedding.
            if R_M > S_M:
                print("\nSTATISTICAL INDICATION: R_M > S_M. The image appears to be a *natural cover* or has minimal embedding.")
            elif S_M > R_M:
                print("\nSTATISTICAL INDICATION: S_M > R_M. This suggests high levels of *random LSB replacement* embedding have occurred.")
            else:
                print("\nSTATISTICAL INDICATION: R_M ≈ S_M. This is consistent with a completely *random LSB layer* (e.g., 50% embedding).")

            return R_M, S_M, R_minus_M, S_minus_M

        except FileNotFoundError:
            print(f"ERROR: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return None


# --- Example Usage ---

if __name__ == '__main__':
    # NOTE: To run this, you need Pillow and numpy installed ('pip install Pillow numpy').
    # A full RS analysis usually requires images and stego-images generated 
    # with known embedding rates to test the intersection point logic.

    analyst = RSAnalysis()

    """
    # 1. Setup a dummy image file
    cover_path = "dummy_cover_for_rs.png"
    try:
        # Create an image that is mostly smooth (25% Red, 75% Blue) to favor R > S initially
        base_array = np.full((100, 100, 3), 100, dtype=np.uint8)
        base_array[:, :25, 0] = 200 # Red spike (adds some edges/non-smoothness)
        base_array[75:, :, 2] = 250 # Blue spike
        dummy_img = Image.fromarray(base_array, 'RGB')
        dummy_img.save(cover_path)
        print(f"Created dummy cover image at: {cover_path}")
    except Exception as e:
        print(f"Could not create dummy image (required PIL/numpy): {e}")

    # 2. Run Analysis
    print("\n--- Starting RS Analysis on the DUMMY COVER Image ---")
    analyst = RSAnalysis()
    analyst.analyze_image(cover_path)
    """

    # 3. Simulate a LSB-embedded image and analyze it (Requires your lsb_sequential script)
    # This section is commented out as it depends on having your other classes/scripts available.
    try:
        # Adds parent directory to Python path
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from steganography.lsb_sequential import LsbSequential
        cover_path = "./Picture1.png"
        stego_path = "./Picture1_stego.png"
        lsb_tool = LsbSequential()

        # Embeds a long random message (simulates high embedding rate)
        long_message = 'A' * 500 
        print("\n--- Simulating 500 character LSB embedding ---")
        lsb_tool.encode_message(cover_path, long_message, stego_path)

        print("\n--- Starting RS Analysis on the STEGO Image ---")
        analyst.analyze_image(stego_path)
    except ImportError:
        print("\nSkipping stego analysis: Cannot import LsbSequential class.")
