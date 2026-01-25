#!/usr/bin/env python3
"""
Sequential LSB Embedding Tool with BPP Control
For testing RS Analysis at different embedding rates
"""

import os
import sys
import numpy as np
from PIL import Image
import random
import argparse


def generate_random_message(length_bits):
    """Generate a random binary message of specified length."""
    return ''.join(str(random.randint(0, 1)) for _ in range(length_bits))


def text_to_binary(text):
    """Convert text to binary string."""
    return ''.join(format(ord(char), '08b') for char in text)


def embed_lsb_sequential(image, message_bits):
    """
    Embed message bits into image using sequential LSB embedding.
    Embeds in order: R0,G0,B0, R1,G1,B1, ... (pixel by pixel, channel by channel)
    """
    img_array = np.array(image, dtype=np.uint8)
    original_shape = img_array.shape
    flat = img_array.flatten().copy()
    
    if len(message_bits) > len(flat):
        raise ValueError(f"Message ({len(message_bits)} bits) exceeds capacity ({len(flat)} bits)")
    
    # Embed each bit
    for i, bit in enumerate(message_bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    
    return Image.fromarray(flat.reshape(original_shape))


def extract_lsb_sequential(image, message_length):
    """Extract message bits from stego image."""
    img_array = np.array(image, dtype=np.uint8)
    flat = img_array.flatten()
    return ''.join(str(flat[i] & 1) for i in range(min(message_length, len(flat))))


def calculate_bits_from_bpp(image, bpp):
    """
    Calculate number of bits to embed based on bits-per-pixel rate.
    
    For 256x256 image = 65,536 pixels
    bpp=0.5 means 32,768 bits embedded
    bpp=1.0 means 65,536 bits embedded (1 bit per pixel on average)
    bpp=3.0 means 196,608 bits (all channels used) - for RGB
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
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image info
    width, height = image.size
    total_pixels = width * height
    total_channels = total_pixels * 3
    
    # Calculate embedding rate
    bpp = len(message_bits) / total_pixels
    percentage = (len(message_bits) / total_channels) * 100
    
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


def main():
    parser = argparse.ArgumentParser(
        description='LSB Embedding with BPP Control for RS Analysis Testing',
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

For 256x256 image (65,536 pixels, 196,608 channels for RGB):

  BPP     | Bits      | % Capacity | RS Detection
  --------|-----------|------------|------------------
  0.05    | 3,277     | 1.67%      | Very Hard
  0.10    | 6,554     | 3.33%      | Hard
  0.25    | 16,384    | 8.33%      | Moderate
  0.50    | 32,768    | 16.67%     | Easier
  1.00    | 65,536    | 33.33%     | Easy
  2.00    | 131,072   | 66.67%     | Very Easy
  3.00    | 196,608   | 100%       | Trivial

═══════════════════════════════════════════════════════════════════
        """
    )
    
    parser.add_argument('-f', '--folder', type=str, required=True,
                        help='Folder containing PNG images')
    
    # Embedding rate options (mutually exclusive)
    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument('--bpp', type=float,
                           help='Bits per pixel (e.g., 0.5, 1.0, 2.0)')
    rate_group.add_argument('--bits', type=int,
                           help='Exact number of bits to embed')
    rate_group.add_argument('--percent', type=float,
                           help='Percentage of capacity to use (0-100)')
    rate_group.add_argument('--multi', action='store_true',
                           help='Generate multiple rates: 0.1, 0.25, 0.5, 1.0, 2.0 bpp')
    
    parser.add_argument('-m', '--message', type=str, default=None,
                        help='Your message (text or binary). If not provided, random bits used.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder name')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification step')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible messages')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Resolve folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        sys.exit(1)
    
    # Get PNG files
    png_files = sorted([f for f in os.listdir(folder_path) 
                        if f.lower().endswith('.png') 
                        and os.path.isfile(os.path.join(folder_path, f))])
    
    if not png_files:
        print(f"No PNG files found in '{folder_path}'")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  LSB EMBEDDING FOR RS ANALYSIS TESTING")
    print(f"{'='*60}")
    print(f"Source folder: {folder_path}")
    print(f"Images found: {len(png_files)}")
    
    # Reference image for calculations
    ref_image = Image.open(os.path.join(folder_path, png_files[0]))
    if ref_image.mode != 'RGB':
        ref_image = ref_image.convert('RGB')
    
    width, height = ref_image.size
    total_pixels = width * height
    total_channels = total_pixels * 3
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Max capacity: {total_channels:,} bits")
    
    # Handle multi-rate mode
    if args.multi:
        bpp_rates = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        print(f"\n>>> MULTI-RATE MODE: Will generate {len(bpp_rates)} versions")
        
        for bpp in bpp_rates:
            num_bits = int(total_pixels * bpp)
            if num_bits > total_channels:
                num_bits = total_channels
                
            output_folder = os.path.join(folder_path, f'stego_bpp_{bpp:.2f}')
            os.makedirs(output_folder, exist_ok=True)
            
            print(f"\n{'─'*60}")
            print(f"Processing BPP = {bpp:.2f} ({num_bits:,} bits)")
            print(f"Output: {output_folder}")
            
            # Generate message for this rate
            if args.message:
                if all(c in '01' for c in args.message):
                    message_bits = args.message
                else:
                    message_bits = text_to_binary(args.message)
                # Pad or truncate
                if len(message_bits) < num_bits:
                    message_bits += generate_random_message(num_bits - len(message_bits))
                else:
                    message_bits = message_bits[:num_bits]
            else:
                message_bits = generate_random_message(num_bits)
            
            # Process each image
            for filename in png_files:
                input_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_folder, filename)
                print(f"\n  [{filename}]")
                process_image(input_path, output_path, message_bits, 
                            verify=not args.no_verify)
        
        print(f"\n{'='*60}")
        print(f"COMPLETE! Created {len(bpp_rates)} embedding rate versions")
        print(f"{'='*60}")
        sys.exit(0)
    
    # Single rate mode - determine number of bits
    if args.bpp is not None:
        num_bits = int(total_pixels * args.bpp)
        rate_desc = f"{args.bpp} bpp"
    elif args.bits is not None:
        num_bits = args.bits
        rate_desc = f"{args.bits} bits"
    elif args.percent is not None:
        num_bits = int(total_channels * (args.percent / 100))
        rate_desc = f"{args.percent}%"
    else:
        # Default to 0.5 bpp
        num_bits = int(total_pixels * 0.5)
        rate_desc = "0.5 bpp (default)"
    
    # Cap at maximum capacity
    if num_bits > total_channels:
        print(f"\nWarning: Requested {num_bits:,} bits exceeds capacity {total_channels:,}")
        num_bits = total_channels
    
    # Calculate actual BPP
    actual_bpp = num_bits / total_pixels
    actual_percent = (num_bits / total_channels) * 100
    
    print(f"\nEmbedding rate: {rate_desc}")
    print(f"  = {num_bits:,} bits")
    print(f"  = {actual_bpp:.4f} bpp")
    print(f"  = {actual_percent:.2f}% of capacity")
    
    # Prepare message
    if args.message:
        print(f"\nUsing your message...")
        if all(c in '01' for c in args.message):
            message_bits = args.message
            print(f"  Detected as binary: {len(message_bits)} bits")
        else:
            message_bits = text_to_binary(args.message)
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
    
    # Output folder
    if args.output:
        output_folder = os.path.join(script_dir, args.output)
    else:
        output_folder = os.path.join(folder_path, f'stego_bpp_{actual_bpp:.2f}')
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")
    
    # Process images
    print(f"\n{'─'*60}")
    print("PROCESSING IMAGES")
    print(f"{'─'*60}")
    
    success = 0
    for filename in png_files:
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"\n[{filename}]")
        if process_image(input_path, output_path, message_bits, 
                        verify=not args.no_verify):
            success += 1
    
    # Save embedding info
    info_path = os.path.join(output_folder, 'embedding_info.txt')
    with open(info_path, 'w') as f:
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


if __name__ == '__main__':
    main()