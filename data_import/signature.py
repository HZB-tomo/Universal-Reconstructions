"""
This file implements HOW a plugin declares "this is my kind of file" --
and it does so with plain, deterministic yes/no checks, not a fuzzy
confidence score. We discussed this at length: confidence scores are
opaque (why did it score 0.72?), while a checklist of required
conditions is fully explainable (it failed because path X was missing).

The core idea, called "specificity": a signature is just a list of
required conditions. The MORE conditions a signature checks, the more
specific it is, and the more we should trust a match from it over a
vaguer one. A vendor plugin that checks 5 exact paths+attributes is
much more likely to be correct than a generic fallback that only checks
"is this a valid hdf5 file at all" (1 condition). See plugin.py and
importer.py for how specificity is used to pick a winner when multiple
plugins match.

Two concrete signature types exist because HDF5 and TIFF are structured
completely differently -- HDF5 has internal named paths and attributes,
while a TIFF "file" is really a folder of many files plus loose sidecar
metadata files. Trying to force both into one generic "signature" shape
would make both harder to use correctly, so we just have two.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class SignatureResult:
    """
    The outcome of checking one signature against one file/folder.
    Carries not just a yes/no answer but WHICH conditions were checked
    and WHICH ones failed -- this is what makes detection debuggable:
    when nothing matches, we can tell the user exactly why, per plugin.
    """

    matched: bool
    checked: list[str]  # human-readable description of every condition tested
    failed: list[str]   # human-readable description of conditions that failed


class Signature(ABC):
    """
    Common interface for both signature types below. Nothing outside
    this file should need to know whether it's dealing with an
    Hdf5Signature or a TiffSignature -- both just answer check() and
    expose a specificity score.
    """

    @abstractmethod
    def check(self, target) -> SignatureResult:
        """
        Test this signature against `target`, which is a format-specific
        handle: an open h5py.File for HDF5 signatures, or a folder Path
        for TIFF signatures.
        """

    @property
    @abstractmethod
    def specificity(self) -> int:
        """
        How many distinct conditions this signature requires. Used as
        the tie-breaker when multiple plugins match the same file --
        higher specificity wins, because it means the plugin "knows
        more" about this exact file structure and is less likely to be
        a coincidental match.
        """


@dataclass
class Hdf5Signature(Signature):
    """
    A signature for HDF5 files: a list of dataset/group paths that must
    exist, plus optionally specific attribute values those paths (or the
    file root) must have.

    Example: a vendor that always stores projections at "/exchange/data"
    with a root attribute "instrument_manufacturer" == "Zeiss" would use:

        Hdf5Signature(
            required_paths=["/exchange/data"],
            required_attrs={"/": {"instrument_manufacturer": "Zeiss"}},
        )
    """

    # Internal HDF5 paths that must all exist for this signature to match.
    required_paths: list[str] = field(default_factory=list)

    # Attribute values that must match exactly. Shape: {path: {attr_name: expected_value}}.
    # Use "/" as the path to check root-level (file-wide) attributes.
    required_attrs: dict[str, dict] = field(default_factory=dict)

    # Escape hatch for conditions that aren't simple path/attribute
    # checks (e.g. "does any dataset have exactly 3 dimensions?"). Takes
    # the open h5py.File and returns (passed: bool, description: str).
    # Optional -- most plugins won't need this.
    custom_predicate: Optional[Callable[[object], tuple[bool, str]]] = None

    def check(self, handle) -> SignatureResult:
        checked: list[str] = []
        failed: list[str] = []

        for path in self.required_paths:
            checked.append(f"path exists: {path}")
            if path not in handle:
                failed.append(f"missing path: {path}")

        for path, attrs in self.required_attrs.items():
            for attr_name, expected in attrs.items():
                checked.append(f"attr {path}@{attr_name} == {expected!r}")
                node = handle.get(path)
                if node is None or node.attrs.get(attr_name) != expected:
                    failed.append(f"attr mismatch: {path}@{attr_name}")

        if self.custom_predicate is not None:
            passed, description = self.custom_predicate(handle)
            checked.append(f"custom: {description}")
            if not passed:
                failed.append(f"custom check failed: {description}")

        return SignatureResult(matched=not failed, checked=checked, failed=failed)

    @property
    def specificity(self) -> int:
        # Count every distinct condition: each required path, each
        # required attribute, and the custom predicate if present.
        count = len(self.required_paths)
        count += sum(len(attrs) for attrs in self.required_attrs.values())
        count += 1 if self.custom_predicate is not None else 0
        return count


@dataclass
class TiffSignature(Signature):
    """
    A signature for TIFF projects: since there's no internal path
    hierarchy to check (it's just a folder of files), conditions here
    are about filename patterns, the presence of sidecar metadata files
    (json/xml/txt), and required keys within those sidecars.

    Example: a vendor that writes "proj_00001.tif" files alongside an
    "acquisition.json" containing an "angles" key would use:

        TiffSignature(
            filename_pattern=r"proj_\\d+\\.tif$",
            required_sidecar_patterns=["acquisition.json"],
            required_sidecar_keys={"acquisition.json": ["angles"]},
        )
    """

    filename_pattern: Optional[str] = None  # regex tested against tiff filenames
    required_sidecar_patterns: list[str] = field(default_factory=list)  # glob patterns, e.g. "*.json"
    required_sidecar_keys: dict[str, list[str]] = field(default_factory=dict)  # {sidecar_glob: [required keys]}
    min_file_count: int = 1  # how many tiff files must be present at minimum

    def check(self, folder: Path) -> SignatureResult:
        import re
        import json

        folder = Path(folder)
        checked: list[str] = []
        failed: list[str] = []

        tiff_files = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in (".tif", ".tiff")
        )
        checked.append(f"tiff file count >= {self.min_file_count}")
        if len(tiff_files) < self.min_file_count:
            failed.append(f"only {len(tiff_files)} tiff files found")

        if self.filename_pattern:
            checked.append(f"filenames match pattern: {self.filename_pattern}")
            pattern = re.compile(self.filename_pattern)
            if not any(pattern.search(p.name) for p in tiff_files):
                failed.append("no filenames matched pattern")

        for pattern_str in self.required_sidecar_patterns:
            checked.append(f"sidecar exists: {pattern_str}")
            if not list(folder.rglob(pattern_str)):
                failed.append(f"missing sidecar: {pattern_str}")

        for pattern_str, required_keys in self.required_sidecar_keys.items():
            matches = list(folder.rglob(pattern_str))
            for key in required_keys:
                checked.append(f"sidecar {pattern_str} has key: {key}")
            if not matches:
                for key in required_keys:
                    failed.append(f"sidecar {pattern_str} not found (needed key {key})")
                continue
            try:
                data = json.loads(matches[0].read_text())
            except Exception:
                data = {}
            for key in required_keys:
                if key not in data:
                    failed.append(f"sidecar {pattern_str} missing key: {key}")

        return SignatureResult(matched=not failed, checked=checked, failed=failed)

    @property
    def specificity(self) -> int:
        count = 1 if self.filename_pattern else 0
        count += len(self.required_sidecar_patterns)
        count += sum(len(keys) for keys in self.required_sidecar_keys.values())
        return count
