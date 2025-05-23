#!/usr/bin/env zsh
set -euo pipefail

# Usage: sync_and_process.sh [--debug]

# TODO support deleting originals once stable

###
# refernece: https://www.rcsb.org/docs/programmatic-access/file-download-services
# script: https://files.wwpdb.org/pub/pdb/software/rsyncPDB.sh

SCRIPT_DIR=${0:A:h}
PYTHON_SCRIPT="$SCRIPT_DIR/process_pdb_files.py"

RAW_PDB_DIR="${HOME}/rcsb_pdb"
TARGET_SUBDIR="structures/divided/pdb"

LOG_RSYNC="${RAW_PDB_DIR}/logs/rsync_$(date +%Y%m%d_%H%M%S).log"
SERVER="rsync.wwpdb.org::ftp"
PORT=33444

###

DEBUG=0
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG=1
  RAW_PDB_DIR="${RAW_PDB_DIR}/_debug"
  echo "DEBUG mode: only fetch first 100 files"
  set -x
fi

mkdir -p "${RAW_PDB_DIR}/${TARGET_SUBDIR}" "$(dirname "$LOG_RSYNC")"

if [[ "$DEBUG" -eq 1 ]]; then
  # build list of first 100 PDB entries
  curl -s 'https://data.rcsb.org/rest/v1/holdings/current/entry_ids' \
  | jq -r '.[]' \
  | sed -n '1,100p' \
  | awk '{ id=tolower($0); dir=substr(id,2,2); print dir "/pdb" id ".ent.gz" }' \
  | sed 's|^|https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/|' \
  > "${RAW_PDB_DIR}/first100_urls.txt" || {
    echo "Failed to fetch first 100 PDB entries"
    exit 1
  }

  # download files not present
  while read -r url; do
    file="${RAW_PDB_DIR}/${TARGET_SUBDIR}/$(basename "$url")"
    if [[ -f $file ]]; then
      echo "Skipping $file"
    else
      curl --silent -fL "$url" -o "$file" || echo "Failed to download $url"
    fi
  done < "${RAW_PDB_DIR}/first100_urls.txt"

  rm -f "${RAW_PDB_DIR}/first100_urls.txt"

else
  echo "Starting sync at $(date) to ${RAW_PDB_DIR}. Log @ ${LOG_RSYNC}"

  # full mirror of divided pdb tree
  rsync -rlptvz --delete --port=${PORT} \
  ${SERVER}/data/${TARGET_SUBDIR}/ ${RAW_PDB_DIR}/${TARGET_SUBDIR}/ \
  > "${LOG_RSYNC}" 2>&1 || {
    echo "Rsync failed; check log at ${LOG_RSYNC}"
    exit 1
  }

  echo "Rsync done."

fi

echo "Processing PDBS"
python3 "${PYTHON_SCRIPT}" --pdb_dir "${RAW_PDB_DIR}/${TARGET_SUBDIR}" --write_dir "${RAW_PDB_DIR}/processed"

echo "Done processing; cleaned up originals. Results in:"
echo "${RAW_PDB_DIR}/processed"

