#!/usr/bin/env python3
"""
Sequential LSB Embedding Tool with BPP Control (Grayscale Version)
For testing RS Analysis at different embedding rates
"""

import os
import sys
import numpy as np
from PIL import Image
import random
import argparse
from pathlib import Path


def generate_random_message(length_bits):
    """Generate a random binary message of specified length."""
    return ''.join(str(random.randint(0, 1)) for _ in range(length_bits))


def text_to_binary(text):
    """Convert text to binary string."""
    return ''.join(format(ord(char), '08b') for char in text)


def embed_lsb_sequential(image, message_bits):
    """
    Embed message bits into image using sequential LSB embedding.
    Embeds in order: pixel by pixel for grayscale
    """
    img_array = np.array(image, dtype=np.uint8)
    original_shape = img_array.shape
    flat = img_array.flatten().copy()
    
    if len(message_bits) > len(flat):
        raise ValueError(f"Message ({len(message_bits)} bits) exceeds capacity ({len(flat)} bits)")
    
    # Embed each bit
    for i, bit in enumerate(message_bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    
    return Image.fromarray(flat.reshape(original_shape), mode='L')


def extract_lsb_sequential(image, message_length):
    """Extract message bits from stego image."""
    img_array = np.array(image, dtype=np.uint8)
    flat = img_array.flatten()
    return ''.join(str(flat[i] & 1) for i in range(min(message_length, len(flat))))


def calculate_bits_from_bpp(image, bpp):
    """
    Calculate number of bits to embed based on bits-per-pixel rate.
    
    For 256x256 grayscale image = 65,536 pixels = 65,536 bits capacity
    bpp=0.5 means 32,768 bits embedded
    bpp=1.0 means 65,536 bits embedded (full capacity)
    """
    width, height = image.size
    total_pixels = width * height
    return int(total_pixels * bpp)


def calculate_embedding_rate(image, message_length):
    """Calculate the bits-per-pixel rate."""
    width, height = image.size
    total_pixels = width * height
    return message_length / total_pixels


def process_image(input_path, output_path, message_bits, verify=True):
    """Process a single image."""
    image = Image.open(input_path)
    
    # Convert to Grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Get image info
    width, height = image.size
    total_pixels = width * height
    
    # Calculate embedding rate
    bpp = len(message_bits) / total_pixels
    percentage = (len(message_bits) / total_pixels) * 100
    
    print(f"  Image size: {width}x{height} ({total_pixels:,} pixels)")
    print(f"  Embedding: {len(message_bits):,} bits")
    print(f"  Rate: {bpp:.4f} bpp ({percentage:.2f}% of capacity)")
    
    # Embed
    stego_image = embed_lsb_sequential(image, message_bits)
    stego_image.save(output_path, 'PNG', compress_level=0)
    
    # Verify
    if verify:
        extracted = extract_lsb_sequential(Image.open(output_path), len(message_bits))
        if extracted == message_bits:
            print("  ✓ Verification PASSED")
        else:
            print("  ✗ Verification FAILED")
            return False
    
    return True


def lsb_sequential_encode(input_folder: Path, output_folder: Path, bpp=None, bits=None, 
                          percent=None, multi=False, message=None, verify=True, seed=None):
    """
    Main function to perform LSB sequential encoding on grayscale images.
    
    Args:
        input_folder: Path to folder containing PNG images
        output_folder: Path to output folder (will be created if it doesn't exist)
        bpp: Bits per pixel (e.g., 0.5, 1.0)
        bits: Exact number of bits to embed
        percent: Percentage of capacity to use (0-100)
        multi: Generate multiple rates (0.05, 0.1, 0.25, 0.5, 0.75, 1.0 bpp)
        message: Message to embed (text or binary). If None, random bits used.
        verify: Whether to verify embedding (default True)
        seed: Random seed for reproducible messages
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Convert to Path objects if strings were passed
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    if not input_folder.exists():
        print(f"Error: Folder '{input_folder}' does not exist!")
        sys.exit(1)
    
    # Get PNG files
    png_files = sorted([f for f in input_folder.iterdir() 
                        if f.suffix.lower() == '.png' and f.is_file()])
    
    if not png_files:
        print(f"No PNG files found in '{input_folder}'")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  LSB EMBEDDING FOR RS ANALYSIS TESTING (GRAYSCALE)")
    print(f"{'='*60}")
    print(f"Source folder: {input_folder}")
    print(f"Images found: {len(png_files)}")
    
    # Reference image for calculations
    ref_image = Image.open(png_files[0])
    if ref_image.mode != 'L':
        ref_image = ref_image.convert('L')
    
    width, height = ref_image.size
    total_pixels = width * height
    max_capacity = total_pixels  # For grayscale, capacity = number of pixels
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Max capacity: {max_capacity:,} bits (grayscale)")
    
    # Handle multi-rate mode
    if multi:
        bpp_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        print(f"\n>>> MULTI-RATE MODE: Will generate {len(bpp_rates)} versions")
        
        for bpp_rate in bpp_rates:
            num_bits = int(total_pixels * bpp_rate)
            if num_bits > max_capacity:
                num_bits = max_capacity
                
            rate_output_folder = output_folder / f'stego_bpp_{bpp_rate:.2f}'
            rate_output_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'─'*60}")
            print(f"Processing BPP = {bpp_rate:.2f} ({num_bits:,} bits)")
            print(f"Output: {rate_output_folder}")
            
            # Generate message for this rate
            if message:
                if all(c in '01' for c in message):
                    message_bits = message
                else:
                    message_bits = text_to_binary(message)
                # Pad or truncate
                if len(message_bits) < num_bits:
                    message_bits += generate_random_message(num_bits - len(message_bits))
                else:
                    message_bits = message_bits[:num_bits]
            else:
                message_bits = generate_random_message(num_bits)
            
            # Process each image
            for png_file in png_files:
                output_path = rate_output_folder / png_file.name
                print(f"\n  [{png_file.name}]")
                process_image(png_file, output_path, message_bits, verify=verify)
        
        print(f"\n{'='*60}")
        print(f"COMPLETE! Created {len(bpp_rates)} embedding rate versions")
        print(f"{'='*60}")
        return
    
    # Single rate mode - determine number of bits
    if bpp is not None:
        num_bits = int(total_pixels * bpp)
        rate_desc = f"{bpp} bpp"
    elif bits is not None:
        num_bits = bits
        rate_desc = f"{bits} bits"
    elif percent is not None:
        num_bits = int(max_capacity * (percent / 100))
        rate_desc = f"{percent}%"
    else:
        # Default to 0.5 bpp
        num_bits = int(total_pixels * 0.5)
        rate_desc = "0.5 bpp (default)"
    
    # Cap at maximum capacity
    if num_bits > max_capacity:
        print(f"\nWarning: Requested {num_bits:,} bits exceeds capacity {max_capacity:,}")
        num_bits = max_capacity
    
    # Calculate actual BPP
    actual_bpp = num_bits / total_pixels
    actual_percent = (num_bits / max_capacity) * 100
    
    print(f"\nEmbedding rate: {rate_desc}")
    print(f"  = {num_bits:,} bits")
    print(f"  = {actual_bpp:.4f} bpp")
    print(f"  = {actual_percent:.2f}% of capacity")
    
    # Prepare message
    if message:
        print(f"\nUsing your message...")
        if all(c in '01' for c in message):
            message_bits = message
            print(f"  Detected as binary: {len(message_bits)} bits")
        else:
            message_bits = text_to_binary(message)
            print(f"  Converted text to binary: {len(message_bits)} bits")
        
        # Pad or truncate to desired length
        if len(message_bits) < num_bits:
            padding = num_bits - len(message_bits)
            message_bits += generate_random_message(padding)
            print(f"  Padded with {padding} random bits to reach {num_bits} bits")
        elif len(message_bits) > num_bits:
            message_bits = message_bits[:num_bits]
            print(f"  Truncated to {num_bits} bits")
    else:
        print(f"\nGenerating random message: {num_bits:,} bits")
        message_bits = generate_random_message(num_bits)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")
    
    # Process images
    print(f"\n{'─'*60}")
    print("PROCESSING IMAGES")
    print(f"{'─'*60}")
    
    success = 0
    for png_file in png_files:
        output_path = output_folder / png_file.name
        
        print(f"\n[{png_file.name}]")
        if process_image(png_file, output_path, message_bits, verify=verify):
            success += 1
    
    # Save embedding info
    info_path = output_folder / 'embedding_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Image Mode: Grayscale\n")
        f.write(f"Embedding Rate: {actual_bpp:.4f} bpp\n")
        f.write(f"Bits Embedded: {num_bits}\n")
        f.write(f"Capacity Used: {actual_percent:.2f}%\n")
        f.write(f"Images Processed: {success}/{len(png_files)}\n")
        f.write(f"\nMessage (first 500 bits):\n{message_bits[:500]}\n")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"  Images processed: {success}/{len(png_files)}")
    print(f"  Output: {output_folder}")
    print(f"  Info saved: embedding_info.txt")
    print(f"{'='*60}\n")


def main():
    """CLI wrapper for lsb_sequential_encode."""
    parser = argparse.ArgumentParser(
        description='LSB Embedding with BPP Control for RS Analysis Testing (Grayscale)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════
                        USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════

1. Embed at specific BITS PER PIXEL rate:
   python lsb_embed.py -f images --bpp 0.5
   python lsb_embed.py -f images --bpp 1.0
   python lsb_embed.py -f images --bpp 0.1

2. Embed specific number of BITS:
   python lsb_embed.py -f images --bits 10000
   python lsb_embed.py -f images --bits 50000

3. Embed at PERCENTAGE of capacity:
   python lsb_embed.py -f images --percent 50
   python lsb_embed.py -f images --percent 25

4. With YOUR OWN MESSAGE (text):
   python lsb_embed.py -f images --bpp 0.5 --message "Your secret message here"

5. With YOUR OWN MESSAGE (binary):
   python lsb_embed.py -f images --bits 16 --message "1010101010101010"

6. Generate MULTIPLE embedding rates for RS analysis:
   python lsb_embed.py -f images --multi

═══════════════════════════════════════════════════════════════════
                     EMBEDDING RATES GUIDE
═══════════════════════════════════════════════════════════════════

For 256x256 grayscale image (65,536 pixels, 65,536 bits capacity):

  BPP     | Bits      | % Capacity | RS Detection
  --------|-----------|------------|------------------
  0.05    | 3,277     | 5%         | Very Hard
  0.10    | 6,554     | 10%        | Hard
  0.25    | 16,384    | 25%        | Moderate
  0.50    | 32,768    | 50%        | Easier
  0.75    | 49,152    | 75%        | Easy
  1.00    | 65,536    | 100%       | Very Easy

═══════════════════════════════════════════════════════════════════
        """
    )
    
    parser.add_argument('-f', '--folder', type=Path, required=True,
                        help='Folder containing PNG images')
    
    # Embedding rate options (mutually exclusive)
    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument('--bpp', type=float,
                           help='Bits per pixel (e.g., 0.5, 1.0)')
    rate_group.add_argument('--bits', type=int,
                           help='Exact number of bits to embed')
    rate_group.add_argument('--percent', type=float,
                           help='Percentage of capacity to use (0-100)')
    rate_group.add_argument('--multi', action='store_true',
                           help='Generate multiple rates: 0.05, 0.1, 0.25, 0.5, 0.75, 1.0 bpp')
    
    parser.add_argument('-m', '--message', type=str, default=None,
                        help='Your message (text or binary). If not provided, random bits used.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output folder path')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification step')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible messages')
    
    args = parser.parse_args()
    
    # Call the main encoding function
    lsb_sequential_encode(
        input_folder=args.folder,
        output_folder=args.output,
        bpp=args.bpp,
        bits=args.bits,
        percent=args.percent,
        multi=args.multi,
        message=args.message,
        verify=not args.no_verify,
        seed=args.seed
    )


if __name__ == '__main__':
    main()