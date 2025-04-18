from enum import Enum


class StrEnum(str, Enum):
    """
    Base enum string class. Nicer printing, better comparisons and dict support.
    """

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        """Compare two instances."""
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        """Return unique hash, so it can be used as a dict key or in a set"""
        return hash(self.value.lower())
