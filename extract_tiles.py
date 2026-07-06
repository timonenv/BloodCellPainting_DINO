"""
python3 extractor.py --help
usage: extractor.py [-h] [-s] [-o] [-b] [--overwrite] INPUT_DIR OUTPUT_DIR

Extract tiles from images in INPUT_DIR and save them to OUTPUT_DIR.

positional arguments:
  INPUT_DIR           directory with *.tiff files
  OUTPUT_DIR          output directory for tile images

optional arguments:
  -h, --help          show this help message and exit
  -s , --size         size of each tile image (default: 224)
  -o , --overlap      overlap between neighbouring tiles (default: 0.25)
  -b , --background   skip tiles with too high black pixel percentage (default: 0.05)
  --overwrite         remove OUTPUT_DIR before extracting tiles (default: False)

python3 extractor.py /data/user/images/ /data/user/tiles/ --overwrite
Removing OUTPUT_DIR='/data/user/tiles/'.
Processing 1 images.
    processing p1_wB2_t1_m1_c2_z1_l1_o0... saved 144 tile images!

BY JOONA POHJONEN
"""

import argparse
import itertools
import os
import shutil
import sys

import numpy as np
from PIL import Image


def extract_tiles_TIFF(
    path: str,
    output_dir: str,
    size: int = 512,
    overlap: float = 0.2,
    background: float = 0.05,
    ) -> None:
    """Extract tiles from image.

    Args:
        path: Path to the tiff/tif image.
        output_dir: Output directory.
        size: Size of each tile image (size x size). Defaults to 512.
        overlap: Overlap between neighbouring tiles. Defaults to 0.2.
        background: Maximum amount of completely black pixels. This happens
            at image edges where the tile goes out of the image. Defaults to
            0.05.
    """
    # Read image.
    image = Image.open(path)

    # Create output dir.
    os.makedirs(output_dir, exist_ok=True)
    # Create output path.
    filename = os.path.basename(path).rstrip(".tiff")
    output_path = os.path.join(
        output_dir,
        filename + "_x{}_y{}_w{}_h{}.jpeg",
    )
    num_images = 0
    for xywh in _tile_coordinates(image, size=size, overlap=overlap):
        # Read region.
        tile = _read_region(image, xywh)
        # Convert to uint8.
        tile = ((np.array(tile) / 2**16) * 255).astype("uint8")
        if (tile == 0).sum() / tile.size > background:
            continue
        # Save.
        Image.fromarray(tile).save(output_path.format(*xywh))
        num_images += 1

def normalize_16bit_PNG(x, newRange=(0, 1)): #x is an array. Default range is between zero and one
    xmin, xmax = np.min(x), np.max(x) #get max and min from input array
    norm = (x - xmin)/(xmax - xmin) # scale between zero and one
    
    if newRange == (0, 1):
        return(norm) # wanted range is the same as norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0] #scale to a different range.    

def extract_tiles_PNG(
    path: str,
    output_dir: str,
    size: int = 512,
    overlap: float = 0.2,
    background: float = 1.0, # cannot lose any tiles
    ) -> None:
    """Extract tiles from image.

    Args:
        path: Path to the tiff/tif image.
        output_dir: Output directory.
        size: Size of each tile image (size x size). Defaults to 512.
        overlap: Overlap between neighbouring tiles. Defaults to 0.2.
        background: Maximum amount of completely black pixels. This happens
            at image edges where the tile goes out of the image. Defaults to
            0.05.
    """
    # Read image.
    image = Image.open(path)
    # Create output dir.
    os.makedirs(output_dir, exist_ok=True)
    # Create output path.
    filename = os.path.basename(path).rstrip(".png")
    output_path = os.path.join(
        output_dir,
        filename + "_x{}_y{}_w{}_h{}.png",
    )

    num_images = 0
    for xywh in _tile_coordinates(image, size=size, overlap=overlap):
        # Read region.
        tile = _read_region(image, xywh)
        # Convert to uint8.
        tile = normalize_16bit_PNG(tile, newRange=(0,2**16)) # TODO added
        tile = ((np.array(tile) / 2**16) * 255).astype("uint8")
        if (tile == 0).sum() / tile.size > background:
            continue
        # Save.
        Image.fromarray(tile).save(output_path.format(*xywh))
        num_images += 1

def extract_tiles_PNG_16bit(
    path: str,
    output_dir: str,
    size: int = 512,
    overlap: float = 0.2,
    background: float = 1.0, # cannot lose any tiles
    ) -> None:
    """Extract tiles from image.

    Args:
        path: Path to the tiff/tif image.
        output_dir: Output directory.
        size: Size of each tile image (size x size). Defaults to 512.
        overlap: Overlap between neighbouring tiles. Defaults to 0.2.
        background: Maximum amount of completely black pixels. This happens
            at image edges where the tile goes out of the image. Defaults to
            0.05.
    """
    # Read image.
    image = Image.open(path)
    # Ensure image is in 16-bit mode
    if image.mode != 'I;16':  # Check if it's not already 16-bit
        image = image.convert('I;16')  # Convert to 16-bit grayscale if not

    # Create output dir.
    os.makedirs(output_dir, exist_ok=True)
    # Create output path.
    filename = os.path.basename(path).rstrip(".png")
    output_path = os.path.join(
        output_dir,
        filename + "_x{}_y{}_w{}_h{}.png",
    )
    #print("    processing {}...".format(filename), end="")
    num_images = 0
    for xywh in _tile_coordinates(image, size=size, overlap=overlap):
        # Read region.
        tile = _read_region(image, xywh)
        # Convert tile to uint16 before performing sum check
        if isinstance(tile, Image.Image):
            tile = np.array(tile, dtype=np.uint16)  # Convert to NumPy array if it's a PIL image
        tile = tile.astype(np.uint16)        
        
        if (tile == 0).sum() / tile.size > background:
            continue
        
        # Check if file already exists before saving
        file_to_save = output_path.format(*xywh)
        if os.path.exists(file_to_save):
            continue

        # Save.
        Image.fromarray(tile.astype(np.uint16)).save(file_to_save, format='PNG', mode='I;16')
        num_images += 1

def _tile_coordinates(image: Image.Image, size: int, overlap: float = 0.0):
    """Returns size x size xywh coordinates for the image."""
    dimensions = image.size
    overlap_px = int(size * overlap)
    # Collect y coords.
    y = [0]
    while y[-1] < dimensions[0]:
        y.append(y[-1] + size - overlap_px)
    y = y[:-1]
    # Collect x coords.
    x = [0]
    while x[-1] < dimensions[1]:
        x.append(x[-1] + size - overlap_px)
    x = x[:-1]
    # Take product.
    coordinates = list(itertools.product(x, y))
    # Add width and height.
    return [xy + (size, size) for xy in coordinates]


def _read_region(image, xywh):
    x, y, w, h = xywh
    return image.crop((x, y, x + w, y + h))


def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract tiles from images in INPUT_DIR and save them to "
            "OUTPUT_DIR."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", metavar="INPUT_DIR", help="directory with *.tiff files"
    )
    parser.add_argument(
        "output", metavar="OUTPUT_DIR", help="output directory for tile images"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=224,
        help="size of each tile image",
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=float,
        default=0.25,
        metavar="",
        help="overlap between neighbouring tiles",
    )
    parser.add_argument(
        "-b",
        "--background",
        type=float,
        default=0.05,
        metavar="",
        help="skip tiles with too high black pixel percentage",
    )
    parser.add_argument(
        "-ch",
        "--channel",
        type=float,
        default=1,
        metavar="",
        help="Which channel to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="remove OUTPUT_DIR before extracting tiles",
    )
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("ERROR: INPUT_DIR doesn't exist.")
        sys.exit(1)
    elif os.path.isfile(args.input):
        print("ERROR: INPUT_DIR exists but isn't a directory.")
        sys.exit(1)
    if os.path.exists(args.output) and os.path.isfile(args.output):
        print("ERROR: OUTPUT_DIR exists but isn't a directory.")
        sys.exit(1)
    if args.overlap < 0 or args.overlap > 1:
        print("ERROR: 'overlap' should be between 0 and 1.")
        sys.exit(1)
    if args.background > 1:
        print("ERROR: 'max_baground' should below 1.")
        sys.exit(1)
    return args


def main(args):
    from datetime import datetime
    startTime = datetime.now()

    # specific to BCP folder structure
    ch1_paths = [
        f.path for f in os.scandir(args.input) if "c1" in f.name and "z0" in f.name
    ch2_paths = [
        f.path for f in os.scandir(args.input) if "c2" in f.name and "z0" in f.name
    ]
    ch3_paths = [
        f.path for f in os.scandir(args.input) if "c3" in f.name and "z0" in f.name
    ]
    ch4_paths = [
        f.path for f in os.scandir(args.input) if "c4" in f.name and "z0" in f.name
    ]
    ch5_paths = [
        f.path for f in os.scandir(args.input) if "c5" in f.name and "z0" in f.name
    ]

    if args.overwrite and os.path.exists(args.output):
        print("Removing OUTPUT_DIR='{}'.".format(args.output))
        shutil.rmtree(args.output)

    print("Processing {} images for ch1.".format(len(ch1_paths)))
    print("Processing {} images for ch2.".format(len(ch2_paths)))
    print("Processing {} images for ch3.".format(len(ch3_paths)))
    print("Processing {} images for ch4.".format(len(ch4_paths)))
    print("Processing {} images for ch5.".format(len(ch5_paths)))
    print("For donor in path: ", args.output)

    for path in ch1_paths:
        extract_tiles_PNG_16bit(
            path,
            output_dir=args.output + "c1",
            size=args.size,
            overlap=args.overlap,
            background=args.background,
        )

    for path in ch2_paths:
        extract_tiles_PNG_16bit(
            path,
            output_dir=args.output + "c2",
            size=args.size,
            overlap=args.overlap,
            background=args.background,
        )
    for path in ch3_paths:
        extract_tiles_PNG_16bit(
            path,
            output_dir=args.output + "c3",
            size=args.size,
            overlap=args.overlap,
            background=args.background,
        )
    for path in ch4_paths:
        extract_tiles_PNG_16bit(
            path,
            output_dir=args.output + "c4",
            size=args.size,
            overlap=args.overlap,
            background=args.background,
        )
    for path in ch5_paths:
        extract_tiles_PNG_16bit(
            path,
            output_dir=args.output + "c5",
            size=args.size,
            overlap=args.overlap,
            background=args.background,
        )
    print(datetime.now() - startTime)


if __name__ == "__main__":
    main(get_arguments())
