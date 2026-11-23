"""Print the dimensions of a video."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Input does not exist: ' + str(args.input))
    from moviepy.video.io.VideoFileClip import VideoFileClip
    with VideoFileClip(str(args.input)) as clip:
        print(f'视频宽度: {clip.w}px')
        print(f'视频高度: {clip.h}px')


if __name__ == '__main__':
    main()
