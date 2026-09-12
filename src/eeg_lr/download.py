"""Download selected public EEGMMIDB imagery runs (not redistributed in Git)."""
import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--subjects', type=int, nargs='+', default=[11])
    args = parser.parse_args()
    if any(s < 1 or s > 109 for s in args.subjects):
        parser.error('Subjects must be between 1 and 109')
    for subject in sorted(set(args.subjects)):
        for run in (4, 8, 12):
            for extension in ('edf', 'edf.event'):
                relative = f'S{subject:03d}/S{subject:03d}R{run:02d}.{extension}'
                target = args.data_dir / relative
                if target.exists():
                    print(f'Exists: {target}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + '.part')
                print(f'Downloading {relative}', flush=True)
                urlretrieve('https://physionet.org/files/eegmmidb/1.0.0/' + relative, temporary)
                temporary.replace(target)


if __name__ == '__main__':
    main()
