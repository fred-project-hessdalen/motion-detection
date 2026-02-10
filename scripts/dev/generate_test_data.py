"""Generate grid of test videos with varying circle size and noise levels."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def generate_test_video(
    output_path: Path,
    circle_radius: int,
    noise_level: int,
    width: int = 640,
    height: int = 480,
    num_frames: int = 30,
) -> None:
    frames: list[NDArray[np.uint8]] = []
    frame: NDArray[np.uint8]
    noise: NDArray[np.uint8]

    # Black frames at start
    for _ in range(40):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if noise_level > 0:
            noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
            frame = np.asarray(cv2.add(frame, noise), dtype=np.uint8)
        frames.append(frame)

    start_x = width // 3
    start_y = height // 2
    end_x = 2 * width // 3
    end_y = height // 2

    # Movement frames
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if noise_level > 0:
            noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
            frame = np.asarray(cv2.add(frame, noise), dtype=np.uint8)

        t = i / (num_frames - 1) if num_frames > 1 else 0
        x = int(start_x + t * (end_x - start_x))
        y = int(start_y + t * (end_y - start_y))

        cv2.circle(frame, (x, y), circle_radius, (255, 255, 255), -1)
        frames.append(frame)

    # Black frames at end
    for _ in range(40):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if noise_level > 0:
            noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
            frame = np.asarray(cv2.add(frame, noise), dtype=np.uint8)
        frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def main() -> None:
    output_dir = Path("data/test_grid")
    output_dir.mkdir(parents=True, exist_ok=True)

    circle_sizes = [1, 5, 10]
    noise_levels = [0, 25, 50]

    for noise in noise_levels:
        for size in circle_sizes:
            video_path = output_dir / f"test_r{size}_n{noise}.mp4"
            generate_test_video(video_path, size, noise)


if __name__ == "__main__":
    main()
