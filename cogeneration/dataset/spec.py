from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cogeneration.util.log import rank_zero_logger

PATH_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = rank_zero_logger(__name__)


@dataclass
class DatasetSpec:
    """
    Paths etc. defining a dataset's metadata and processed data.
    """

    # name of dataset
    name: str
    # path to metadata CSV, which contains all the information about the dataset
    metadata_path: Path
    # directory where processed data is stored, esp if relative paths defined in CSV
    processed_root_path: Path
    # cluster file
    cluster_path: Optional[Path] = None

    def __post_init__(self):
        try:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata CSV for {self.name} not found: {self.metadata_path}"
                )
            if not self.processed_root_path.exists():
                raise FileNotFoundError(
                    f"Processed data root path for {self.name} not found: {self.processed_root_path}"
                )
            if self.cluster_path and not self.cluster_path.exists():
                raise FileNotFoundError(
                    f"Cluster file for {self.name} not found: {self.cluster_path}"
                )
            self._enabled = True
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Dataset {self.name} is disabled: {e}")
            self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled


# cogeneration data pipeline generated datasets
# TODO - generate shared clusters file for these datasets (but they arent really used for anything...)

cogeneration_datasets_path = Path("~/pdb/").expanduser().resolve()

CogenerationPDBDatasetSpec = DatasetSpec(
    name="CogenerationPDB",
    processed_root_path=cogeneration_datasets_path,
    metadata_path=cogeneration_datasets_path / "rcsb" / "processed" / "metadata.csv",
)

CogenerationAFDBDatasetSpec = DatasetSpec(
    name="CogenerationAFDB",
    processed_root_path=cogeneration_datasets_path,
    metadata_path=cogeneration_datasets_path
    / "alphafold"
    / "processed"
    / "metadata.csv",
)

CogenerationRedesignDatasetSpec = DatasetSpec(
    name="CogenerationRedesigns",
    processed_root_path=cogeneration_datasets_path,
    metadata_path=cogeneration_datasets_path / "redesigned" / "redesigns.csv",
)

# MultiFlow `BestRedesigns` CSV is refolded to generate a new redesign dataset
MultiflowRedesignedDatasetSpec = DatasetSpec(
    name="MultiflowRedesigns",
    processed_root_path=cogeneration_datasets_path / "redesigned" / "multiflow",
    metadata_path=cogeneration_datasets_path
    / "redesigned"
    / "multiflow"
    / "redesigns.csv",
)

# multiflow metadata paths

multiflow_datasets_path = PATH_PROJECT_ROOT / "cogeneration" / "datasets"
multiflow_metadata = multiflow_datasets_path / "multiflow"

MultiflowSyntheticDatasetSpec = DatasetSpec(
    name="MultiFlowSynthetic",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "distillation_metadata.csv",
    cluster_path=multiflow_metadata / "distillation.clusters",
)


# NOTE - Multiflow PDB datasets are superseded by CogenerationPDBDatasetSpec

MultiflowPDBDatasetSpec = DatasetSpec(
    name="MultiflowPDB",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "pdb_metadata.csv",
    cluster_path=multiflow_metadata / "pdb.clusters",
)

MultiflowPDBTestDatasetSpec = DatasetSpec(
    name="MultiflowPDBPost2021",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "test_set_metadata.csv",
    cluster_path=multiflow_metadata / "test_set_clusters.csv",
)
