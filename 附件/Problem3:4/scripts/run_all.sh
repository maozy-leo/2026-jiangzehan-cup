#!/bin/bash
# Parallel runner for all cities under multiple configs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_CMD="${PY_CMD:-python}"
export PYTHONPATH="$ROOT_DIR"

# Editable lists -------------------------------------------------------------
cities=(Chengdu Dalian Dongguan Harbin Qingdao Quanzhou Shenyang Zhengzhou)
configs=(config_adaptive.json config_weights.json)
radii=(100 300 500)

# Limit concurrent processes to avoid overwhelming the machine.
MAX_PROCS="${MAX_PROCS:-16}"

wait_for_slot() {
  while (( "$(jobs -pr | wc -l)" >= MAX_PROCS )); do
    wait -n || true
  done
}

for cfg in "${configs[@]}"; do
  cfg_path="$ROOT_DIR/configs/$cfg"
  cfg_name="${cfg%.json}"
  output_root="$ROOT_DIR/outputs/$cfg_name"
  mkdir -p "$output_root"

  for radius in "${radii[@]}"; do
    radius_dir="$output_root/radius_${radius}m"
    mkdir -p "$radius_dir"

    for city in "${cities[@]}"; do
      csv="$ROOT_DIR/data/${city}_Edgelist.csv"
      if [[ ! -f "$csv" ]]; then
        echo "[warn] missing CSV for $city ($csv), skipping" >&2
        continue
      fi
      wait_for_slot
      (
        set -x
        "$PY_CMD" -m src.runner "$city" "$csv" \
          --config "$cfg_path" \
          --output-dir "$radius_dir" \
          --attack-mode "radius" \
          --radius-m "$radius"
      ) &
    done
  done
done

wait
echo "All runs completed."
