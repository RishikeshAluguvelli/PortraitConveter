"""Calculate static crop region to keep person in frame."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..utils.logger import logger
from ..utils.config import Config


class CropCalculator:
    """Calculate optimal crop region for portrait video."""

    def __init__(self):
        """Initialize crop calculator."""
        self.output_width = Config.OUTPUT_WIDTH
        self.output_height = Config.OUTPUT_HEIGHT
        self.aspect_ratio = Config.OUTPUT_ASPECT_RATIO

    def calculate_static_crop(self, video_path: str, target_person_detections: List[Dict],
                             start_frame: int = 0, end_frame: int = -1) -> Optional[Tuple[int, int, int, int]]:
        """Calculate a single static crop region that keeps eyes/nose in frame for all frames.

        Args:
            video_path: Path to video file
            target_person_detections: List of person detections (one per frame)
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            (x, y, width, height) crop region or None
        """
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Collect all eye and nose positions
        eye_centers = []
        noses = []

        for detection in target_person_detections:
            if detection is None:
                continue

            # Get nose position
            nose = detection.get('nose')
            if nose:
                noses.append(nose)

            # Get eye center
            left_eye = detection.get('left_eye')
            right_eye = detection.get('right_eye')

            if left_eye and right_eye:
                eye_center = (
                    (left_eye[0] + right_eye[0]) / 2.0,
                    (left_eye[1] + right_eye[1]) / 2.0
                )
                eye_centers.append(eye_center)

        if not eye_centers and not noses:
            logger.warning("No valid eye/nose positions found")
            return None

        # Use eye centers primarily, fallback to noses
        reference_points = eye_centers if eye_centers else noses

        # Find bounding box that contains all reference points
        all_x = [pt[0] for pt in reference_points]
        all_y = [pt[1] for pt in reference_points]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Calculate center point (use float for precision)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        logger.info(f"Reference points: {len(reference_points)} points")
        logger.info(f"X range: {min_x} to {max_x}, Y range: {min_y} to {max_y}")
        logger.info(f"Center point: ({center_x:.1f}, {center_y:.1f})")

        # Calculate crop dimensions based on 9:16 aspect ratio
        crop_height = video_height
        crop_width = int(crop_height * self.aspect_ratio)

        # If crop width exceeds video width, adjust
        if crop_width > video_width:
            crop_width = video_width
            crop_height = int(crop_width / self.aspect_ratio)

        logger.info(f"Video dimensions: {video_width}x{video_height}")
        logger.info(f"Crop dimensions: {crop_width}x{crop_height}")

        # Position crop to keep eyes in upper portion of frame
        # Eyes should be at about 30% from top of the crop
        # So: crop_y + (crop_height * 0.3) = center_y
        # Therefore: crop_y = center_y - (crop_height * 0.3)
        crop_y = int(center_y - crop_height * Config.CROP_PADDING_TOP)
        crop_x = int(center_x - crop_width / 2.0)

        # Ensure crop stays within video bounds
        crop_x = max(0, min(crop_x, video_width - crop_width))
        crop_y = max(0, min(crop_y, video_height - crop_height))

        # Verify all reference points are within crop
        # If not, expand crop area
        for pt in reference_points:
            px, py = pt

            # Check if point is outside crop region
            if px < crop_x or px > crop_x + crop_width or py < crop_y or py > crop_y + crop_height:
                # Recalculate with more padding
                logger.info("Adjusting crop to ensure all keypoints are in frame")
                return self._calculate_expanded_crop(
                    video_width, video_height, reference_points
                )

        logger.info(f"Calculated static crop: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}")

        return (crop_x, crop_y, crop_width, crop_height)

    def calculate_center_crop(self, video_path: str) -> Tuple[int, int, int, int]:
        """Calculate a center crop for the video (fallback when no persons detected).

        Args:
            video_path: Path to video file

        Returns:
            (x, y, width, height) crop region
        """
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate crop dimensions based on 9:16 aspect ratio
        crop_height = video_height
        crop_width = int(crop_height * self.aspect_ratio)

        # If crop width exceeds video width, adjust
        if crop_width > video_width:
            crop_width = video_width
            crop_height = int(crop_width / self.aspect_ratio)

        # Center the crop
        crop_x = int((video_width - crop_width) / 2.0)
        crop_y = int((video_height - crop_height) / 2.0)

        logger.info(f"Calculated center crop: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}")

        return (crop_x, crop_y, crop_width, crop_height)

    def _calculate_expanded_crop(self, video_width: int, video_height: int,
                                reference_points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Calculate expanded crop when standard crop doesn't fit all points.

        Args:
            video_width: Video width
            video_height: Video height
            reference_points: List of (x, y) points to keep in frame

        Returns:
            (x, y, width, height) crop region
        """
        # Find bounding box with padding
        all_x = [pt[0] for pt in reference_points]
        all_y = [pt[1] for pt in reference_points]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Add padding
        padding_x = int((max_x - min_x) * 0.5)
        padding_y = int((max_y - min_y) * 0.5)

        min_x = max(0, min_x - padding_x)
        max_x = min(video_width, max_x + padding_x)
        min_y = max(0, min_y - padding_y)
        max_y = min(video_height, max_y + padding_y)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        # Calculate dimensions
        crop_height = video_height
        crop_width = int(crop_height * self.aspect_ratio)

        if crop_width > video_width:
            crop_width = video_width
            crop_height = int(crop_width / self.aspect_ratio)

        # Center on reference points
        crop_x = int(center_x - crop_width / 2.0)
        crop_y = int(center_y - crop_height / 2.0)

        # Clamp to bounds
        crop_x = max(0, min(crop_x, video_width - crop_width))
        crop_y = max(0, min(crop_y, video_height - crop_height))

        return (crop_x, crop_y, crop_width, crop_height)

    def apply_crop(self, video_path: str, output_path: str, crop_region: Tuple[int, int, int, int],
                  start_frame: int = 0, end_frame: int = -1) -> bool:
        """Apply crop to video and save.

        Args:
            video_path: Input video path
            output_path: Output video path
            crop_region: (x, y, width, height) crop region
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == -1:
            end_frame = total_frames

        crop_x, crop_y, crop_width, crop_height = crop_region

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.output_width, self.output_height)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_written = 0
        for frame_num in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame
            cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            # Resize to output dimensions
            resized = cv2.resize(cropped, (self.output_width, self.output_height))

            out.write(resized)
            frames_written += 1

        cap.release()
        out.release()

        logger.info(f"Cropped {frames_written} frames to {output_path}")

        return frames_written > 0
