from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from memex.engine.sources import Source, SourceFile, register_source


@register_source
class LocalSource(Source):
    """Source backed by a local filesystem directory."""

    type = "local"

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = ext.strip().lower()
        return f".{ext}" if not ext.startswith(".") else ext

    def __init__(
        self,
        name: str,
        path: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> None:
        self.name = name
        self.extensions = [self._normalize_ext(e) for e in (extensions or [])]
        self.recursive = recursive
        self._root = Path(path).resolve()

    def list_files(self) -> list[SourceFile]:
        """List all files under the root directory matching extensions."""
        if not self._root.is_dir():
            return []

        pattern = "**/*" if self.recursive else "*"
        result: list[SourceFile] = []

        for p in self._root.glob(pattern):
            if not p.is_file():
                continue
            if self.extensions and p.suffix.lower() not in self.extensions:
                continue
            st = p.stat()
            result.append(
                SourceFile(
                    name=p.name,
                    path=str(p),
                    size=st.st_size,
                    modified_at=st.st_mtime,
                )
            )

        return result

    def get_content_hash(self, file: SourceFile) -> str:
        """Compute SHA-256 hash of file contents."""
        h = hashlib.sha256()
        with open(file.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def download(self, file: SourceFile, dest: Path) -> Path:
        """Copy the local file to *dest* and return the target path."""
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / file.name
        shutil.copy2(file.path, target)
        return target
