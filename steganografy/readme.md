

# LSB Methods

| Method | desc |
| ---- | --- |
| Sequential LSB Replacement | Read image pixels, directly manipulate the LSB of the pixel values e.g., using bitwise operations like & | 
| Randomized LSB Embedding | Use a cryptographic key as the seed for a Pseudo-Random Number Generator (PRNG). The PRNG output sequence determines which pixel locations are selected for embedding the secret message, instead of just using them sequentially. |
| LSB Matching ($\pm 1$ Embedding) | Implement the core logic: if the message bit doesn't match the pixel's LSB, randomly add +1 or -1 to the pixel value. This requires manual pixel-wise logic using Python/Numpy. |
| Adaptive Steganography | This requires calculating a "complexity" or "texture" map (e.g., based on edge detection like Canny or local variance) for the cover image. Only pixels in high-complexity regions are selected for embedding. |

<br/>


## Randomized LSB Embedding

The key to randomized LSB is the use of a secret key (often a password or a number) to initialize a Pseudo-Random Number Generator (PRNG).

1. PRNG and Key: The key is used as the seed for the PRNG. When the same key (seed) is used, the PRNG generates the exact same, reproducible sequence of "random" numbers every single time.

2. Path Generation: This random sequence of numbers is used to determine the exact, non-sequential order of pixels (or pixel channels) in the image where the secret bits will be embedded. This creates a secret, diffused path.

3. Security: An attacker without the correct key will not only fail to know if a message exists, but they will also be unable to follow the correct path to extract the bits in the right order, resulting in garbled, unintelligible data.

The algorithm uses the standard LSB replacement technique, but instead of iterating (0,0), (0,1), (0,2)... it jumps around the image based on the seeded sequence.


##  LSB Matching ($\pm 1$ Embedding)

**Understanding LSB Matching ($\pm 1$ Embedding)**

Standard LSB Replacement introduces a detectable statistical bias: if you need to change the LSB of a pixel, the pixel value is always moved closer to the value of the bit you are embedding. This non-random pattern is easily detected by modern steganalysis techniques (like the RS analysis).

**LSB Matching solves this by aiming for a $\mathbf{0.5}$ modification rate (the minimum possible changes):**

1. Check for Match: Compare the Secret Bit ($M$) with the Least Significant Bit of the Cover Pixel ($P_{LSB}$).

2. No Change (50% of the time): If $M = P_{LSB}$, no modification is performed. The pixel remains $P$.

3. Required Change (50% of the time): If $M \neq P_{LSB}$, a change is necessary. The standard replacement would just flip the LSB (e.g., $25 \to 24$ or $24 \to 25$). LSB Matching instead does the following:
    * Randomly choose to add $+1$ or $-1$ to the pixel value $P$.
    * Both $(P+1)$ and $(P-1)$ will result in the new LSB equaling the secret bit $M$.

The Advantage: By randomly selecting between $+1$ and $-1$, the algorithm maintains the statistical histogram of the cover image much better than simple replacement, making the stego-image far more resistant to statistical attacks.


## Adaptive Steganography (The Edge Detection method)

**How the Edge Detection Works in Adaptive LSB Steganography**

The script uses a simplified edge detection mechanism to identify pixels that are suitable for hiding data, thus implementing an adaptive approach to minimize visual artifacts.

1. Complexity Calculation: Inside the _get_edge_mask method, a simplified version of the Sobel edge detection operator is used. It works by calculating the absolute difference between neighboring pixels (approximating horizontal and vertical gradients) in the Red channel.

2. Thresholding: The sum of these differences (the gradient magnitude) represents the local complexity or texture of the image area. This magnitude is used to score how "busy" a pixel neighborhood is.

3. Adaptive Selection: Only pixels where this magnitude is greater than the instance's defined threshold (default is 35 in the example) are marked as True in the edge_mask. These are the selected "edge" or "high-complexity" regions.

4. Embedding: The encode_message method then iterates through the pixels and only modifies the Least Significant Bit (LSB) of the color channels that correspond to a True value in the edge_mask. This ensures the subtle changes introduced by LSB replacement are masked by the image's existing high-frequency details.

5. Synchronization (Key to Decoding): For successful data retrieval, the decode_message method must use the exact same _get_edge_mask logic and threshold value to correctly re-identify the sequence of embedded pixels. This synchronization is crucial for extracting the hidden message bit-for-bit.

