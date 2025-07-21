import datetime as dt
from typing import Any, Mapping, Optional, Set

import pandas as pd
import torch
from Bio.PDB.Structure import Structure

from cogeneration.type.str_enum import StrEnum


def extract_structure_date(structure: Structure) -> Optional[dt.date]:
    """
    Extract the date of structure deposition from the Biopython Structure header.
    """
    header: Mapping[str, str] = getattr(structure, "header", {})

    for key in (
        "deposition_date",  # normalised datetime.date
        "deposition_date_original",  # raw string (older Biopython)
        "release_date",  # mmCIF key
    ):
        date_val = header.get(key)
        if isinstance(date_val, dt.date):
            return date_val
        if isinstance(date_val, str):
            try:
                return dt.datetime.strptime(date_val.strip(), "%Y-%m-%d").date()
            except ValueError:
                pass  # keep trying

    # fallback to HEADER, line format: cols 50-59 => DD-MON-YY (e.g. 12-APR-89)
    header_line = next(
        (l for l in header.get("raw_header_lines", []) if l.startswith("HEADER")),
        None,
    )
    if header_line:
        raw = header_line[50:59].strip()
        try:
            return dt.datetime.strptime(raw, "%d-%b-%y").date()
        except ValueError:
            pass

    return None


class StructureExperimentalMethod(StrEnum):
    """
    Structure experimental / modeling methods, ~ aligned with Boltz-2
    """

    MD = "MD"
    XRAY_DIFFRACTION = "XRAY_DIFFRACTION"
    ELECTRON_MICROSCOPY = "ELECTRON_MICROSCOPY"
    SOLUTION_NMR = "SOLUTION_NMR"
    OTHER_EXPERIMENTAL = "OTHER"  # bucket for many less-common methods
    # synthetic
    AFDB = "AFDB"  # AF2 DB
    BOLTZ_1 = "BOLTZ-1"  # Boltz-1 structures
    # placeholder for future methods... safe-ish to change enum values?
    FUTURE1 = "FUTURE1"
    FUTURE2 = "FUTURE2"
    FUTURE3 = "FUTURE3"
    FUTURE4 = "FUTURE4"
    FUTURE5 = "FUTURE5"

    @classmethod
    def default_tensor_feat(cls) -> torch.Tensor:
        """Returns a (1, ) tensor with the default method (XRAY_DIFFRACTION)."""
        return cls.to_tensor(cls.XRAY_DIFFRACTION)

    @staticmethod
    def to_tensor(value: "StructureExperimentalMethod") -> torch.Tensor:
        """Return StructureExperimentalMethod enum as long."""
        return torch.tensor(
            [StructureExperimentalMethod.to_int(value)], dtype=torch.long
        )

    @staticmethod
    def to_int(value: "StructureExperimentalMethod") -> int:
        """Return StructureExperimentalMethod enum as int."""
        for i, method in enumerate(StructureExperimentalMethod):
            if method == value:
                return i
        raise ValueError(f"Unknown structure method: {value}")

    @staticmethod
    def is_experimental(value: "StructureExperimentalMethod") -> bool:
        """
        Check if the method is an experimental method (not synthetic).
        """
        return value in {
            StructureExperimentalMethod.MD,
            StructureExperimentalMethod.XRAY_DIFFRACTION,
            StructureExperimentalMethod.ELECTRON_MICROSCOPY,
            StructureExperimentalMethod.SOLUTION_NMR,
            StructureExperimentalMethod.OTHER_EXPERIMENTAL,
        }

    @classmethod
    def from_value(cls, value: Any) -> "StructureExperimentalMethod":
        """
        Infer method from a string value, which may be enum value or structure header value.
        """
        if isinstance(value, cls):
            return value

        # handle missing in CSV as default to X-RAY DIFFRACTION
        if pd.isna(value):
            return cls.XRAY_DIFFRACTION

        _STR_TO_METHOD: Mapping[str, StructureExperimentalMethod] = {
            "MD": cls.MD,
            "X-RAY DIFFRACTION": cls.XRAY_DIFFRACTION,
            "ELECTRON MICROSCOPY": cls.ELECTRON_MICROSCOPY,
            "SOLUTION NMR": cls.SOLUTION_NMR,
            # less common methods binned to OTHER_EXPERIMENTAL
            "SOLID-STATE NMR": cls.OTHER_EXPERIMENTAL,
            "NEUTRON DIFFRACTION": cls.OTHER_EXPERIMENTAL,
            "ELECTRON CRYSTALLOGRAPHY": cls.OTHER_EXPERIMENTAL,
            "FIBER DIFFRACTION": cls.OTHER_EXPERIMENTAL,
            "POWDER DIFFRACTION": cls.OTHER_EXPERIMENTAL,
            "INFRARED SPECTROSCOPY": cls.OTHER_EXPERIMENTAL,
            "FLUORESCENCE TRANSFER": cls.OTHER_EXPERIMENTAL,
            "EPR": cls.OTHER_EXPERIMENTAL,
            "THEORETICAL MODEL": cls.OTHER_EXPERIMENTAL,
            "SOLUTION SCATTERING": cls.OTHER_EXPERIMENTAL,
            "OTHER": cls.OTHER_EXPERIMENTAL,
            # synthetic
            "AFDB": cls.AFDB,
            "BOLTZ-1": cls.BOLTZ_1,
            # future
            "FUTURE1": cls.FUTURE1,
            "FUTURE2": cls.FUTURE2,
            "FUTURE3": cls.FUTURE3,
            "FUTURE4": cls.FUTURE4,
            "FUTURE5": cls.FUTURE5,
        }

        if not isinstance(value, str):
            raise ValueError(f"Method {value} is not a string")

        # if value is already an enum value, return it
        for key, enum_val in _STR_TO_METHOD.items():
            if value == enum_val:
                return enum_val

        # otherwise, try to match it to a known string
        value = value.upper()
        if value in _STR_TO_METHOD:
            return _STR_TO_METHOD[value]

        raise ValueError(f"Unknown structure method: {value}")

    @classmethod
    def from_structure(cls, structure: Structure) -> "StructureExperimentalMethod":
        """
        Infer method from a Biopython Structure header.
        """
        header: Mapping[str, str] = getattr(structure, "header", {})

        # Check for AlphaFold prediction
        name = (header.get("name") or header.get("title") or "").upper()
        if name and "ALPHAFOLD" in name and "PREDICTION" in name:
            return cls.AFDB

        # PDBParser sets 'structure_method', MMCIFParser may set 'experiment_method'
        raw: str = (
            header.get("structure_method") or header.get("experiment_method") or ""
        ).upper()

        # split on ';' because methods can come concatenated
        for part in map(str.strip, raw.split(";")):
            if not part:
                continue
            try:
                return cls.from_value(part)
            except ValueError:
                # continue to try next part
                continue

        # Don't fallback to an Unknown / other
        raise ValueError(f"Unknown structure method: {raw}")
