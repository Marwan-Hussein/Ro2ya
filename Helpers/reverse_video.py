from pathlib import Path
from moviepy import VideoFileClip, vfx

# Define directory paths relative to this script
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "scroll-left"
OUTPUT_DIR = BASE_DIR / "scroll-right"


# Common video extensions to search for
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def main():
    # Make sure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all video files in Folder-A
    video_files = [
        f
        for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {INPUT_DIR}")
        return

    print(f"Found {len(video_files)} video(s) to process.\n")

    for video_path in video_files:
        output_path = OUTPUT_DIR / video_path.name
        print(f"Reversing: {video_path.name}...")

        try:
            # Load clip
            clip = VideoFileClip(str(video_path))

            # Apply time mirror effect in MoviePy v2.x syntax
            reversed_clip = clip.with_effects([vfx.TimeMirror()])

            # Save reversed video
            reversed_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,  # Suppress progress bar output
            )

            # Close clips to free memory
            clip.close()
            reversed_clip.close()

            print(f"  Saved -> {output_path.name}")

        except Exception as e:
            print(f"  Failed to process {video_path.name}: {e}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
