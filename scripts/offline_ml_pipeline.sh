#!/usr/bin/env bash
set -e

###############################################
#  Auto-detect PROJECT_ROOT
###############################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="python3"

###############################################
#  Hard-coded components
###############################################
COMPONENTS="amplitude phase"

###############################################
#  Parse CLI arguments
###############################################
usage() {
    echo "Usage: $0 --bbh-spin <value> --dataset <value> --mode <value> --model-name <value>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bbh-spin)
            BBH_SPIN="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -*|--*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            break
            ;;
    esac
done

# Validate required flags
if [[ -z "$BBH_SPIN" || -z "$DATASET" || -z "$MODE" || -z "$MODEL_NAME" ]]; then
    usage
fi

###############################################
#  Pipeline execution
###############################################
echo "PROJECT_ROOT detected as: $PROJECT_ROOT"
echo "Running pipeline with:"
echo "  bbh-spin:   $BBH_SPIN"
echo "  dataset:    $DATASET"
echo "  mode:       $MODE"
echo "  components: $COMPONENTS"
echo "  model-name: $MODEL_NAME"
echo ""

for COMP in $COMPONENTS; do
    echo "=== train_ann: component=$COMP ==="
    $PYTHON "$PROJECT_ROOT/scripts/train_ann.py" \
        --bbh-spin "$BBH_SPIN" \
        --dataset "$DATASET" \
        --mode "$MODE" \
        --component "$COMP" \
        --model-name "$MODEL_NAME" \
        --saving --verbose
    echo ""
done

echo "=== finalise_mode_rom ==="
$PYTHON "$PROJECT_ROOT/scripts/finalise_mode_rom.py" \
    --bbh-spin "$BBH_SPIN" \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --model-name "$MODEL_NAME" \
    --verbose
echo ""

echo "=== Pipeline complete ==="
echo ""
