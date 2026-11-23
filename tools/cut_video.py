"""Cut a time range from a video without model inference."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True, help='start time in seconds')
    parser.add_argument('--end', type=float, required=True, help='end time in seconds')
    args = parser.parse_args()
    if not args.input.is_file() or not 0 <= args.start < args.end:
        parser.error('Input must exist and times must satisfy 0 <= start < end.')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files.')
    from moviepy.video.io.VideoFileClip import VideoFileClip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(args.input)) as clip:
        if args.end > clip.duration:
            parser.error('--end exceeds the input duration.')
        subclip = getattr(clip, 'subclipped', None) or clip.subclip
        subclip(args.start, args.end).write_videofile(str(args.output))


if __name__ == '__main__':
    main()
