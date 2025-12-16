#!/usr/bin/env python3
"""Main entry point for Portrait Converter."""

import argparse
import sys
import os

from src.pipeline import PortraitConverter
from src.utils.config import Config
from src.utils.logger import logger


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert landscape podcast videos to portrait format (9:16)"
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input landscape video path"
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output portrait video path"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves detection frames)"
    )

    parser.add_argument(
        "--pose-threshold",
        type=float,
        default=0.50,
        help="YOLO pose detection threshold (default: 0.80)"
    )

    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.95,
        help="RetinaFace detection threshold (default: 0.95)"
    )

    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=20.0,
        help="Scene detection sensitivity (default: 27.0)"
    )

    parser.add_argument(
        "--output-width",
        type=int,
        default=1080,
        help="Output video width (default: 1080)"
    )

    parser.add_argument(
        "--output-height",
        type=int,
        default=1920,
        help="Output video height (default: 1920)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Configure
    Config.set_thresholds(args.pose_threshold, args.face_threshold)
    Config.SCENE_THRESHOLD = args.scene_threshold
    Config.OUTPUT_WIDTH = args.output_width
    Config.OUTPUT_HEIGHT = args.output_height

    logger.info("Portrait Converter")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Pose threshold: {Config.POSE_THRESHOLD}")
    logger.info(f"Face threshold: {Config.FACE_THRESHOLD}")
    logger.info(f"Scene threshold: {Config.SCENE_THRESHOLD}")
    logger.info(f"Output size: {Config.OUTPUT_WIDTH}x{Config.OUTPUT_HEIGHT}")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info("=" * 60)

    # Initialize converter
    converter = PortraitConverter(debug_mode=args.debug)

    # Convert
    success = converter.convert(args.input, args.output)

    if success:
        logger.info("Conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
