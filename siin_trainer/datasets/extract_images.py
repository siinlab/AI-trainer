"""
This module provides functionality to extract frames from video files.

Functions:
    extract_frames_from_video(video_path: str, output_dir: str):
        Extracts frames from a video file and saves them as images in the specified directory.
"""

import os
from pathlib import Path
import cv2
from tqdm import tqdm
from lgg import logger

import torch
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def are_similar_frames(
    frame1: np.ndarray, frame2: np.ndarray, similarity_threshold: float
) -> bool:
    """Determines if two frames are similar based on their CLIP embeddings.

    Args:
        frame1: The first frame (image) to compare.
        frame2: The second frame (image) to compare.
        similarity_threshold: The threshold for considering frames as similar.

    Returns:
        True if the frames are similar, False otherwise.
    """
    # Load and preprocess images
    image1 = preprocess(Image.fromarray(frame1).convert("RGB")).unsqueeze(0).to(device)
    image2 = preprocess(Image.fromarray(frame2).convert("RGB")).unsqueeze(0).to(device)

    # Get image embeddings
    with torch.no_grad():
        emb1 = model.encode_image(image1)
        emb2 = model.encode_image(image2)

    # Normalize embeddings
    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (emb1 @ emb2.T).item()

    return similarity > similarity_threshold


def extract_frames_from_video(
    video_path: Path, output_dir: Path, similarity_threshold: float = 0.5
):
    """
    Extracts frames from a video file and saves them as images in the specified directory.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Path to the directory where frames will be saved.
        similarity_threshold (float): Threshold for considering frames as similar.

    Returns:
        None
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    logger.info(f"Extracting frames from video: {video_path}")

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening video file: {video_path}")
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in video: {frame_count}")

    frame_index = 0
    previous_frame = None
    with tqdm(
        total=frame_count, desc="Extracting frames", unit="frame"
    ) as progress_bar:
        while True:
            progress_bar.update(1)
            # Read the next frame from the video
            success, frame = video_capture.read()
            if not success:
                break
            if previous_frame is not None:
                # Check if the current frame is similar to the previous frame
                if are_similar_frames(previous_frame, frame, similarity_threshold):
                    logger.debug("Skipping similar frame")
                    continue
            frame_filename = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(frame_filename, frame)
            previous_frame = frame

            frame_index += 1

    video_capture.release()
    logger.info(f"Frames extracted and saved to {output_dir}")
