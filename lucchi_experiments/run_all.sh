#!/usr/bin/env bash
# Runs the Lucchi experiments one after another. For each image size and each model:
#   1. config_<model>.ini: training, then testing at the same image size;
#   2. config_<model>_test_256x256x160.ini (if it exists): testing of the same model at 256x256x160.
# The test at 256x256x160 is skipped if the training failed. A failed run does not stop the other models.
#
# Usage (from anywhere):
#   ./lucchi_experiments/run_all.sh                        # all image sizes
#   ./lucchi_experiments/run_all.sh 64x64x40               # one image size
#   ./lucchi_experiments/run_all.sh 64x64x40 128x128x80    # several image sizes

set -o pipefail  # The exit status of "python ... | tee" is the one of python

cd "$(dirname "$0")/.." || exit 1  # Repository root, as the paths in the configs are relative to it

# Activates the virtual environment if needed
if [ -z "$VIRTUAL_ENV" ] && [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Prevents the Mac from sleeping while this script runs
if command -v caffeinate > /dev/null; then
    caffeinate -i -w $$ &
fi

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(64x64x40 128x128x80 256x256x160)
fi
MODELS=(hnoseg_xs hartleymha fnoseg hnoseg fno vnet-ds)
CONFIG_DIR=lucchi_experiments/config_files
LOG_DIR=outputs/logs
mkdir -p "$LOG_DIR"

failed=()

run() {  # $1: config file. The output is shown and saved to $LOG_DIR.
    local log="$LOG_DIR/$(basename "$(dirname "$1")")_$(basename "$1" .ini).log"
    echo
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $1"
    if ! python3 lucchi_experiments/run.py "$1" 2>&1 | tee "$log"; then
        echo "=== FAILED: $1 (see $log)"
        failed+=("$1")
        return 1
    fi
}

for size in "${SIZES[@]}"; do
    if [ ! -d "$CONFIG_DIR/$size" ]; then
        echo "Unknown image size: $size (no folder $CONFIG_DIR/$size)"
        failed+=("$size")
        continue
    fi
    for model in "${MODELS[@]}"; do
        run "$CONFIG_DIR/$size/config_$model.ini" || continue  # No test without a trained model
        test_config="$CONFIG_DIR/$size/config_${model}_test_256x256x160.ini"
        if [ -f "$test_config" ]; then
            run "$test_config"
        fi
    done
done

echo
echo "=== Finished at $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All runs succeeded."
else
    echo "Failed runs:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
