from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PATH_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
    # BestRedesigns CSV (MultiFlow style). Always replaces original sequence with redesign sequence.
    best_redesigns_path: Optional[Path] = None
    # cluster file
    cluster_path: Optional[Path] = None

    def __post_init__(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV for {self.name} not found: {self.metadata_path}"
            )
        if not self.processed_root_path.exists():
            raise FileNotFoundError(
                f"Processed data root path for {self.name} not found: {self.processed_root_path}"
            )
        if self.best_redesigns_path and not self.best_redesigns_path.exists():
            raise FileNotFoundError(
                f"BestRedesigns CSV for {self.name} not found: {self.best_redesigns_path}"
            )
        if self.cluster_path and not self.cluster_path.exists():
            raise FileNotFoundError(
                f"Cluster file for {self.name} not found: {self.cluster_path}"
            )


# cogeneration data pipeline generated datasets

cogeneration_datasets_path = Path("~/pdb/").expanduser().resolve()

CogenerationPDBDatasetSpec = DatasetSpec(
    name="CogenerationPDB",
    processed_root_path=cogeneration_datasets_path,
    metadata_path=cogeneration_datasets_path / "rcsb" / "processed" / "metadata.csv",
    # cluster_path=cogeneration_datasets_path / "rcsb" / "processed" / "cogeneration.clusters",  # TODO?
)

CogenerationAFDBDatasetSpec = DatasetSpec(
    name="CogenerationAFDB",
    processed_root_path=cogeneration_datasets_path,
    metadata_path=cogeneration_datasets_path
    / "alphafold"
    / "processed"
    / "metadata.csv",
    # cluster_path=cogeneration_datasets_path / "alphafold" / "processed" / "cogeneration.clusters",  # TODO?
)

# multiflow metadata paths

multiflow_datasets_path = PATH_PROJECT_ROOT / "cogeneration" / "datasets"
multiflow_metadata = multiflow_datasets_path / "multiflow"

MultiflowPDBDatasetSpec = DatasetSpec(
    name="MultiflowPDB",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "pdb_metadata.csv",
    cluster_path=multiflow_metadata / "pdb.clusters",
)

MultiflowPDBRedesignedDatasetSpec = DatasetSpec(
    name="MultiflowPDBRedesigned",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "pdb_metadata.csv",
    best_redesigns_path=multiflow_metadata / "pdb_redesigned.csv",
    cluster_path=multiflow_metadata / "pdb.clusters",
)

MultiflowPDBTestDatasetSpec = DatasetSpec(
    name="MultiflowPDBPost2021",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "test_set_metadata.csv",
    cluster_path=multiflow_metadata / "test_set_clusters.csv",
)

MultiflowSyntheticDatasetSpec = DatasetSpec(
    name="MultiFlowSynthetic",
    processed_root_path=multiflow_datasets_path,
    metadata_path=multiflow_metadata / "distillation_metadata.csv",
    cluster_path=multiflow_metadata / "distillation.clusters",
)
