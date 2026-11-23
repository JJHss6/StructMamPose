"""Convert a video to a GIF."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=0.03, help='GIF frame duration in seconds')
    args = parser.parse_args()
    if not args.input.is_file() or args.duration <= 0:
        parser.error('Input must exist and frame duration must be positive.')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files.')
    import imageio.v2 as imageio
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_reader(str(args.input), 'ffmpeg') as video:
        frames = [frame for frame in video]
    imageio.mimsave(str(args.output), frames, duration=args.duration)


if __name__ == '__main__':
    main()
