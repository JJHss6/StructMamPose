"""Resize a video, with an optional time range."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--start', type=float, default=0, help='start time in seconds')
    parser.add_argument('--end', type=float, default=None, help='optional end time in seconds')
    args = parser.parse_args()
    if not args.input.is_file() or min(args.width, args.height) <= 0 or args.start < 0:
        parser.error('Input must exist, dimensions must be positive and start must be nonnegative.')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files.')
    from moviepy.video.io.VideoFileClip import VideoFileClip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(args.input)) as clip:
        end = clip.duration if args.end is None else args.end
        if not args.start < end <= clip.duration:
            parser.error('Times must satisfy start < end <= input duration.')
        resize = getattr(clip, 'resized', None) or clip.resize
        resized = resize((args.width, args.height))
        if args.start != 0 or args.end is not None:
            subclip = getattr(resized, 'subclipped', None) or resized.subclip
            resized = subclip(args.start, end)
        resized.write_videofile(str(args.output))


if __name__ == '__main__':
    main()
