import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(entry):
    name = entry['file']
    if not re.fullmatch(r'[A-Za-z0-9_.+\-]+\.whl', name):
        raise ValueError(f'Invalid wheel filename: {name}')
    if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
        raise ValueError(f'Invalid SHA256: {name}')
    if not isinstance(entry['bytes'], int) or entry['bytes'] <= 0:
        raise ValueError(f'Invalid size: {name}')
    parsed = urlparse(entry['url'])
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f'Expected public HTTPS URL: {name}')


def verified(path, entry):
    return (path.is_file() and path.stat().st_size == entry['bytes']
            and sha256(path) == entry['sha256'])


def download(entry, destination, attempts=3, opener=urlopen):
    validate(entry)
    target = destination / entry['file']
    if target.is_symlink():
        raise ValueError(f'Refusing symlink: {target}')
    if verified(target, entry):
        print(f"Cached: {entry['file']}", flush=True)
        return 'cached'
    # Exclusive creation also prevents two invocations sharing a partial file.
    partial = destination / (entry['file'] + f'.{os.getpid()}.part')
    for attempt in range(attempts):
        try:
            request = Request(entry['url'], headers={'User-Agent': 'TempFit-environment/1'})
            with partial.open('xb') as output, opener(request, timeout=120) as response:
                size = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > entry['bytes']:
                        raise ValueError('Response exceeds expected size')
                    output.write(block)
            if not verified(partial, entry):
                raise ValueError('Downloaded wheel size or SHA256 mismatch')
            partial.replace(target)
            print(f"Downloaded: {entry['file']}", flush=True)
            return 'downloaded'
        except FileExistsError:
            raise RuntimeError(f'Stale/concurrent partial file; inspect before removing: {partial}')
        except Exception as error:
            partial.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise RuntimeError(f"Failed: {entry['file']}: {error}") from error
            print(f"Retry {attempt + 1}/{attempts}: {entry['file']}: {error}", flush=True)
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description='Download and hash-check the fixed TempFit environment.')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('DOWNLOAD_MANIFEST.json'))
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['files']
    for entry in entries:
        validate(entry)
    if len({entry['file'] for entry in entries}) != len(entries):
        raise ValueError('Duplicate filenames in manifest')
    args.destination.mkdir(parents=True, exist_ok=True)
    print(f"Verifying/downloading {len(entries)} wheels, {sum(e['bytes'] for e in entries):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(lambda e: download(e, args.destination), entries))
    print(json.dumps({'verified': len(result), 'downloaded': result.count('downloaded'),
                      'cached': result.count('cached')}))


if __name__ == '__main__':
    main()
