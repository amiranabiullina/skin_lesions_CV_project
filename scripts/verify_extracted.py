"""Verify an already-extracted ISIC folder against the official zip without re-downloading it.

Browsers (Safari) often unpack the zip and delete it, so `unzip -t` is impossible. Instead we read
only the zip's central directory from the server (HTTP Range requests, ~1 MB) and compare every
extracted file's size and CRC32 with the official values.

    python scripts/verify_extracted.py ~/Downloads/ISIC2018_Task1-2_Training_Input
"""
import io
import sys
import zipfile
import zlib
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/"


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests (enough for zipfile.infolist)."""

    def __init__(self, url):
        self.url = url
        self.size = int(requests.head(url, timeout=60).headers["content-length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}, timeout=60)
        r.raise_for_status()
        self.pos = end
        return r.content


def crc32(path):
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    return crc


def main(folder):
    folder = Path(folder).expanduser()
    zf = zipfile.ZipFile(HTTPRangeFile(BASE_URL + folder.name + ".zip"))
    members = [i for i in zf.infolist() if not i.is_dir()]
    print(f"official zip lists {len(members)} files")
    bad = []
    for info in tqdm(members):
        path = folder.parent / info.filename
        if not path.exists():
            bad.append((info.filename, "missing"))
        elif path.stat().st_size != info.file_size:
            bad.append((info.filename, f"size {path.stat().st_size} != {info.file_size}"))
        elif crc32(path) != info.CRC:
            bad.append((info.filename, "CRC mismatch"))
    if bad:
        print(f"{len(bad)} problems, e.g. {bad[:5]}")
        sys.exit(1)
    print(f"all {len(members)} files match the official sizes and CRC32")


if __name__ == "__main__":
    main(sys.argv[1])
