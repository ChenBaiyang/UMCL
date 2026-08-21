from __future__ import annotations

import fnmatch
import shutil
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _normalize_relpath(value: str | Path) -> str:
    text = str(value).replace("\\", "/").strip("/")
    if text in {".", ""}:
        return ""
    return text


@dataclass
class _ZipIndex:
    files: set[str]
    directories: set[str]
    children: dict[str, list[str]]


class ZipArchiveStore:
    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path)
        self._zip = zipfile.ZipFile(self.archive_path, mode="r")
        self._index = self._build_index(self._zip.namelist())
        self.root = ZipArchivePath(self, "")

    @staticmethod
    def _build_index(names: list[str]) -> _ZipIndex:
        files: set[str] = set()
        directories: set[str] = {""}
        children: dict[str, set[str]] = defaultdict(set)

        for raw_name in names:
            name = _normalize_relpath(raw_name)
            if not name:
                continue
            if raw_name.endswith("/"):
                directories.add(name)
            else:
                files.add(name)
            parts = name.split("/")
            for depth in range(len(parts)):
                parent = "/".join(parts[:depth])
                child = parts[depth]
                children[parent].add(child)
                if depth + 1 < len(parts):
                    directories.add("/".join(parts[: depth + 1]))

        return _ZipIndex(
            files=files,
            directories=directories,
            children={key: sorted(value) for key, value in children.items()},
        )

    def close(self) -> None:
        self._zip.close()

    def exists(self, relpath: str) -> bool:
        return relpath == "" or relpath in self._index.files or relpath in self._index.directories

    def is_file(self, relpath: str) -> bool:
        return relpath in self._index.files

    def is_dir(self, relpath: str) -> bool:
        return relpath == "" or relpath in self._index.directories

    def children(self, relpath: str) -> list[str]:
        return self._index.children.get(relpath, [])

    def open_bytes(self, relpath: str):
        if not self.is_file(relpath):
            raise FileNotFoundError(f"Missing file inside archive: {relpath}")
        return self._zip.open(relpath, mode="r")

    def read_bytes(self, relpath: str) -> bytes:
        with self.open_bytes(relpath) as handle:
            return handle.read()

    def read_text(self, relpath: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(relpath).decode(encoding)


class ZipArchivePath:
    def __init__(self, store: ZipArchiveStore, relpath: str | Path) -> None:
        self._store = store
        self._relpath = _normalize_relpath(relpath)

    @property
    def parts(self) -> tuple[str, ...]:
        if not self._relpath:
            return ()
        return tuple(self._relpath.split("/"))

    @property
    def name(self) -> str:
        if not self._relpath:
            return self._store.archive_path.stem
        return self._relpath.split("/")[-1]

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix

    @property
    def parent(self) -> "ZipArchivePath":
        if not self._relpath:
            return self
        parent = self._relpath.rsplit("/", 1)[0] if "/" in self._relpath else ""
        return ZipArchivePath(self._store, parent)

    def joinpath(self, *parts: str | Path) -> "ZipArchivePath":
        rel = self._relpath
        for part in parts:
            chunk = _normalize_relpath(part)
            if not chunk:
                continue
            rel = f"{rel}/{chunk}" if rel else chunk
        return ZipArchivePath(self._store, rel)

    def __truediv__(self, other: str | Path) -> "ZipArchivePath":
        return self.joinpath(other)

    def exists(self) -> bool:
        return self._store.exists(self._relpath)

    def is_dir(self) -> bool:
        return self._store.is_dir(self._relpath)

    def is_file(self) -> bool:
        return self._store.is_file(self._relpath)

    def iterdir(self) -> Iterator["ZipArchivePath"]:
        if not self.is_dir():
            raise NotADirectoryError(str(self))
        for child in self._store.children(self._relpath):
            yield ZipArchivePath(self._store, f"{self._relpath}/{child}" if self._relpath else child)

    def glob(self, pattern: str) -> Iterator["ZipArchivePath"]:
        if not self.is_dir():
            return iter(())
        matched = [
            child
            for child in self.iterdir()
            if fnmatch.fnmatchcase(child.name, pattern) and child.is_file()
        ]
        return iter(sorted(matched))

    def open(self, mode: str = "r", encoding: str = "utf-8", newline: str | None = None):
        if "w" in mode or "a" in mode or "+" in mode:
            raise ValueError("ZipArchivePath is read-only")
        handle = self._store.open_bytes(self._relpath)
        if "b" in mode:
            return handle
        import io

        return io.TextIOWrapper(handle, encoding=encoding, newline=newline)

    def read_bytes(self) -> bytes:
        return self._store.read_bytes(self._relpath)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._store.read_text(self._relpath, encoding=encoding)

    def __fspath__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self._relpath:
            return f"{self._store.archive_path}"
        return f"{self._store.archive_path}!/{self._relpath}"

    def __repr__(self) -> str:
        return f"ZipArchivePath({str(self)!r})"

    def __lt__(self, other: object) -> bool:
        return str(self) < str(other)


@contextmanager
def open_download_root(downloads_root: str | Path, dataset_id: str):
    root = Path(downloads_root)
    archive_path = root / f"{dataset_id}.zip"
    dataset_dir = root / dataset_id
    if archive_path.exists():
        store = ZipArchiveStore(archive_path)
        try:
            yield store.root
        finally:
            store.close()
        return
    if dataset_dir.exists():
        yield dataset_dir
        return
    raise FileNotFoundError(
        f"Missing raw download root for {dataset_id!r}: expected {archive_path} or {dataset_dir}"
    )


def build_download_archive(dataset_id: str, downloads_root: str | Path, *, overwrite: bool = False) -> Path:
    root = Path(downloads_root)
    source_dir = root / dataset_id
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source download directory: {source_dir}")
    archive_path = root / f"{dataset_id}.zip"
    if archive_path.exists() and not overwrite:
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir).as_posix()
            if any(part.startswith("._") or part == ".DS_Store" for part in path.parts):
                continue
            zf.write(path, rel)
    return archive_path


def prune_download_source(dataset_id: str, downloads_root: str | Path) -> None:
    root = Path(downloads_root)
    source_dir = root / dataset_id
    if source_dir.exists():
        shutil.rmtree(source_dir)
