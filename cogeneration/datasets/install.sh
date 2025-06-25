# download datasets from multiflow
# https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9

# TODO(dataset) - support downloading other metadata, e.g. Scope already processed
# https://github.com/microsoft/protein-frame-flow/blob/20293ab04994a5bfe97f2d8cf8603a0b63af0609/metadata/scope_metadata.csv

# Ensure we are in the directory containing this file - we want to download datasets here.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Download datasets
wget https://zenodo.org/records/10714631/files/real_train_set.tar.gz
wget https://zenodo.org/records/10714631/files/synthetic_train_set.tar.gz
wget https://zenodo.org/records/10714631/files/test_set.tar.gz

# Uncompress training data
mkdir train_set
tar -xzf real_train_set.tar.gz -C train_set/
tar -xzf synthetic_train_set.tar.gz -C train_set/

# Uncompress test data
mkdir test_set
tar -xzf test_set.tar.gz -C test_set/

# Clean up dataset tarballs
rm real_train_set.tar.gz synthetic_train_set.tar.gz test_set.tar.gz