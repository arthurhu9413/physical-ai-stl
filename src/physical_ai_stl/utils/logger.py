from __future__ import annotations

"""Lightweight, robust CSV experiment logger.

Design goals
------------
- **Safe header handling**: can predefine a header, infer it from an on‑disk file,
  or infer it from the first mapping that is appended. If the file already
  exists, we adopt its header instead of clobbering the file.
- **Concurrent friendly**: optional cross‑platform file lock using a sidecar
  ``.lock`` file ensures multiple processes append safely.
- **Practical ergonomics**: accepts either sequences or mappings; supports
  graceful handling of ``None`` values; optional compact float formatting.
- **Small + dependency‑free**: standard library only and ~250 lines of code.

This module intentionally keeps the API minimal: create a :class:`CSVLogger`
and call :meth:`append`, :meth:`append_many`, or :meth:`extend_header`.

Example
-------
>>> log = CSVLogger('metrics.csv', header=['step', 'loss'])
>>> log.append({'step': 1, 'loss': 0.1234})
>>> log.append([2, 0.0567])  # sequences work too

"""

import csv
import os
import time
import numbers
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["CSVLogger"]


# ---- Internal: a tiny cross‑platform lock using a sidecar ".lock" file ----


class _FileLock:
    """A simple inter‑process file lock using exclusive creation of ``.lock``.

    This avoids platform‑specific fcntl/msvcrt details and works anywhere the
    filesystem supports atomic O_EXCL on create.
    """

    def __init__(self, target: Path, timeout: float = 5.0, poll: float = 0.01) -> None:
        self._lock_path = Path(str(target) + ".lock")
        self._timeout = float(timeout)
        self._poll = float(poll)
        self._fd: int | None = None

    def acquire(self) -> None:
        deadline = time.time() + self._timeout
        while True:
            try:
                # O_EXCL ensures we fail if the file already exists.
                self._fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self._lock_path}")
                time.sleep(self._poll)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        try:
            os.unlink(self._lock_path)
        except FileNotFoundError:
            # Someone may have cleaned up for us; that's fine.
            pass

    def __enter__(self) -> _FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class _NullContext:
    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False


class _Opts:
    __slots__ = (
        "delimiter",
        "encoding",
        "float_precision",
        "none_as_empty",
        "strict_lengths",
        "lock",
        "lock_timeout",
    )

    def __init__(
        self,
        delimiter: str = ",",
        encoding: str = "utf-8",
        float_precision: int | None = None,
        none_as_empty: bool = True,
        strict_lengths: bool = True,
        lock: bool = False,
        lock_timeout: float = 5.0,
    ):
        self.delimiter = delimiter
        self.encoding = encoding
        self.float_precision = float_precision
        self.none_as_empty = none_as_empty
        self.strict_lengths = strict_lengths
        self.lock = lock
        self.lock_timeout = lock_timeout


class CSVLogger:
    """Append‑only CSV logger for small/medium experiment logs.

    Parameters
    ----------
    path:
        Destination CSV file.
    header:
        Optional header (column names). If provided and ``overwrite`` is true (default),
        the file will be clobbered and this header written. If ``overwrite`` is false
        and a file already exists with a different header, a ``ValueError`` is raised.
        If ``header`` is omitted and the file already exists, the on‑disk header (if any)
        is adopted automatically. If ``header`` is omitted and the first appended row is a
        mapping, its keys become the header.
    overwrite:
        Whether to clobber the file when ``header`` is provided. Defaults to True when
        ``header`` is not None, otherwise ignored.
    create_dirs:
        Create parent directories as needed.
    delimiter, encoding:
        CSV formatting options.
    float_precision:
        If given, numeric (non‑integral) values are formatted with ``.{precision}g``.
    none_as_empty:
        If True (default), ``None`` values are written as empty strings rather than "None".
    strict_lengths:
        When a header is set, enforce exact row length. If False, rows are truncated or
        right‑padded with empties.
    lock, lock_timeout:
        Enable/parameterize the optional sidecar lock.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        header: Iterable[str] | None = None,
        *,
        overwrite: bool | None = None,
        create_dirs: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
        float_precision: int | None = None,
        none_as_empty: bool = True,
        strict_lengths: bool = True,
        lock: bool = False,
        lock_timeout: float = 5.0,
    ) -> None:
        self.path = Path(path)
        if create_dirs:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # Basic validation for delimiter/header to fail fast with helpful messages
        if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in {"\n", "\r"}:
            raise ValueError("delimiter must be a single non‑newline character")
        if header is not None:
            header_list = list(header)
            if len(set(header_list)) != len(header_list):
                raise ValueError("header contains duplicate column names")
            if any(not isinstance(c, str) or c == "" for c in header_list):
                raise ValueError("header must be non‑empty strings")
        else:
            header_list = None

        # Store opts compactly
        self._opt = _Opts(
            delimiter=delimiter,
            encoding=encoding,
            float_precision=float_precision,
            none_as_empty=none_as_empty,
            strict_lengths=strict_lengths,
            lock=lock,
            lock_timeout=lock_timeout,
        )
        self._header: list[str] | None = header_list
        self._lock = _FileLock(self.path, timeout=lock_timeout) if lock else None

        # If header is given, mimic prior behavior (write header, clobber file)
        if self._header is not None:
            if overwrite is None:
                overwrite = True
            if overwrite:
                with self._maybe_locked(), self._open("w") as f:
                    csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)
            else:
                if self.path.exists():
                    existing = self._peek_first_line()
                    target = self._opt.delimiter.join(self._header)
                    if existing and existing != target:
                        raise ValueError(
                            "Existing CSV header does not match requested header.\n"
                            f"  Existing: {existing}\n  New:      {target}\n"
                            "Pass overwrite=True to replace the file or adjust the header."
                        )
                else:
                    with self._maybe_locked(), self._open("w") as f:
                        csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)

    # ---- Public API -----------------------------------------------------

    @property
    def header(self) -> tuple[str, ...] | None:
        """Current header as an immutable tuple or ``None`` if not set."""
        return tuple(self._header) if self._header is not None else None

    def append(self, row: Iterable[Any] | Mapping[str, Any]) -> None:
        """Append a single row.

        Accepts either a sequence (list/tuple) or a mapping (dict). For mappings,
        keys must be a subset of the header. If no header is known yet, the
        logger will (1) adopt an on‑disk header if the file already exists and
        has a header, otherwise (2) infer it from the mapping's keys.
        """
        # Make sure we adopt an existing on‑disk header if present.
        self._ensure_header_loaded()

        if isinstance(row, Mapping):
            values = self._row_from_mapping(row)
        else:
            values = self._row_from_sequence(row)

        with self._maybe_locked(), self._open("a") as f:
            csv.writer(f, delimiter=self._opt.delimiter).writerow(values)

    def append_many(self, rows: Iterable[Iterable[Any] | Mapping[str, Any]]) -> None:
        """Append many rows efficiently."""
        # Make sure we adopt an existing on‑disk header if present.
        self._ensure_header_loaded()

        with self._maybe_locked(), self._open("a") as f:
            w = csv.writer(f, delimiter=self._opt.delimiter)
            for row in rows:
                if isinstance(row, Mapping):
                    values = self._row_from_mapping(row)
                else:
                    values = self._row_from_sequence(row)
                w.writerow(values)

    def extend_header(self, new_columns: Iterable[str]) -> None:
        """Extend the header by adding any columns from ``new_columns`` not already present.

        Existing rows are padded with empty strings for the new columns.
        """
        additions: list[str]

        # Single critical section to avoid races with writers
        with self._maybe_locked():
            if self._header is None:
                self._header = list(new_columns)
                with self._open("w") as f:
                    csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)
                return

            additions = [c for c in new_columns if c not in self._header]
            if not additions:
                return

            # Read everything (small logs typical for experiments)
            rows: list[list[str]] = []
            if self.path.exists() and self.path.stat().st_size > 0:
                with self._open("r") as f:
                    reader = list(csv.reader(f, delimiter=self._opt.delimiter))
                rows = reader[1:] if reader else []

            self._header.extend(additions)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", newline="", encoding=self._opt.encoding) as f:
                w = csv.writer(f, delimiter=self._opt.delimiter)
                w.writerow(self._header)
                pad = [""] * len(additions)
                for r in rows:
                    w.writerow(list(r) + pad)
            os.replace(tmp, self.path)

    # ---- Internals ------------------------------------------------------

    def _format_value(self, v: Any) -> Any:
        """Normalize a value for CSV writing (precision + None handling).

        - ``None`` → "" (if ``none_as_empty``).
        - Numeric non‑integral values are optionally formatted using ``g``.
        """
        if v is None and self._opt.none_as_empty:
            return ""
        if self._opt.float_precision is not None and isinstance(v, numbers.Real) and not isinstance(v, numbers.Integral):
            # Cast to float to ensure consistent formatting across numpy/Decimal reals.
            return format(float(v), f".{self._opt.float_precision}g")
        return v

    def _row_from_sequence(self, row: Iterable[Any]) -> list[Any]:
        vals = [self._format_value(v) for v in list(row)]

        if self._header is not None:
            if self._opt.strict_lengths and len(vals) != len(self._header):
                raise ValueError(f"Row length {len(vals)} != header length {len(self._header)}")
            if not self._opt.strict_lengths and len(vals) < len(self._header):
                # Right‑pad with empties
                vals = vals + [""] * (len(self._header) - len(vals))
            elif not self._opt.strict_lengths and len(vals) > len(self._header):
                vals = vals[: len(self._header)]
        return vals

    def _row_from_mapping(self, m: Mapping[str, Any]) -> list[Any]:
        # Adopt existing header if available; otherwise infer from mapping the first time.
        if self._header is None:
            disk_header = self._read_existing_header()
            if disk_header:
                self._header = disk_header
            else:
                self._header = list(m.keys())
                with self._maybe_locked(), self._open("w") as f:
                    csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)

        # Validate keys
        extra = [k for k in m.keys() if k not in self._header]
        if extra:
            raise KeyError(
                f"Mapping contains keys not in header: {extra}. "
                "Call extend_header([...]) to add new columns."
            )

        vals = [self._format_value(m.get(k, "")) for k in self._header]
        return vals

    # --- Header discovery helpers ---------------------------------------

    def _peek_first_line(self) -> str:
        try:
            with self._open("r") as f:
                return f.readline().rstrip("\r\n")
        except FileNotFoundError:
            return ""

    def _read_existing_header(self) -> list[str] | None:
        """Read and parse the first CSV row as header if the file exists and is non‑empty."""
        try:
            with self._open("r") as f:
                r = csv.reader(f, delimiter=self._opt.delimiter)
                first = next(r, None)
            return first if first else None
        except FileNotFoundError:
            return None

    def _ensure_header_loaded(self) -> None:
        """If we don't have a header yet but the file on disk does, adopt it."""
        if self._header is None:
            disk = self._read_existing_header()
            if disk:
                self._header = disk

    # --- File I/O helpers ------------------------------------------------

    def _open(self, mode: str):
        return self.path.open(mode, newline="", encoding=self._opt.encoding)

    def _maybe_locked(self):
        return self._lock if self._lock is not None else _NullContext()
