#!/usr/bin/env bash
# Fetch the three EU regulations from EUR-Lex into ./pdfs.
# Idempotent: skips files that already exist.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${ROOT}/pdfs"
mkdir -p "${PDF_DIR}"

declare -a CORPUS=(
  "DORA.pdf|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554"
  "NIS2.pdf|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2555"
  "AI_Act.pdf|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689"
)

for entry in "${CORPUS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  out="${PDF_DIR}/${name}"
  if [[ -s "${out}" ]]; then
    echo "skip ${name} (already present)"
    continue
  fi
  echo "fetch ${name}"
  curl -fL --retry 3 --retry-delay 2 -A "Mozilla/5.0 lexRAG/0.1" -o "${out}" "${url}"
done

echo "done -> ${PDF_DIR}"
ls -lh "${PDF_DIR}"
