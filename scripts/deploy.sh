#!/bin/bash
set -euo pipefail

# Sentinel Platform Deploy Script
# Reads configuration from .env and applies Kubernetes manifests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load .env file
ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    log_info "Loading configuration from .env"
    set -a
    source "$ENV_FILE"
    set +a
else
    log_error ".env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

# Validate required variables
if [[ -z "${PROJECT_ID:-}" ]]; then
    log_error "PROJECT_ID is not set in .env"
    exit 1
fi

# Default values
ENVIRONMENT="${ENVIRONMENT:-dev}"
GCP_REGION="${GCP_REGION:-us-central1}"

log_info "Configuration:"
log_info "  PROJECT_ID: $PROJECT_ID"
log_info "  ENVIRONMENT: $ENVIRONMENT"
log_info "  GCP_REGION: $GCP_REGION"

# Create temp directory for processed manifests
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to process templates
process_templates() {
    local src_dir="$1"
    local dest_dir="$2"

    mkdir -p "$dest_dir"

    # Copy and substitute environment variables
    find "$src_dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) | while read -r file; do
        rel_path="${file#$src_dir/}"
        dest_file="$dest_dir/$rel_path"
        mkdir -p "$(dirname "$dest_file")"
        envsubst '${PROJECT_ID} ${ENVIRONMENT} ${GCP_REGION}' < "$file" > "$dest_file"
    done

    # Copy non-template files
    find "$src_dir" -type f ! \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) | while read -r file; do
        rel_path="${file#$src_dir/}"
        dest_file="$dest_dir/$rel_path"
        mkdir -p "$(dirname "$dest_file")"
        cp "$file" "$dest_file"
    done
}

# Main deployment
case "${1:-}" in
    "k8s"|"kubernetes")
        log_info "Processing Kubernetes manifests..."
        process_templates "$ROOT_DIR/k8s" "$TEMP_DIR/k8s"

        OVERLAY="$TEMP_DIR/k8s/overlays/$ENVIRONMENT"
        if [[ ! -d "$OVERLAY" ]]; then
            log_error "Overlay not found: $OVERLAY"
            exit 1
        fi

        if [[ "${2:-}" == "--dry-run" ]]; then
            log_info "Dry run - validating manifests..."
            kubectl apply --dry-run=client -k "$OVERLAY"
        else
            log_info "Applying Kubernetes manifests..."
            kubectl apply -k "$OVERLAY"
        fi
        ;;

    "observability")
        log_info "Processing Observability manifests..."
        process_templates "$ROOT_DIR/observability" "$TEMP_DIR/observability"

        if [[ "${2:-}" == "--dry-run" ]]; then
            log_info "Dry run - validating manifests..."
            kubectl apply --dry-run=client -f "$TEMP_DIR/observability/prometheus/"
            kubectl apply --dry-run=client -f "$TEMP_DIR/observability/alertmanager/"
            kubectl apply --dry-run=client -f "$TEMP_DIR/observability/grafana/"
        else
            log_info "Applying Observability manifests..."
            kubectl apply -f "$TEMP_DIR/observability/prometheus/"
            kubectl apply -f "$TEMP_DIR/observability/alertmanager/"
            kubectl apply -f "$TEMP_DIR/observability/grafana/"
        fi
        ;;

    "all")
        log_info "Deploying everything..."
        "$0" k8s "${2:-}"
        "$0" observability "${2:-}"
        ;;

    "render")
        # Just render the templates without applying
        log_info "Rendering templates to ./rendered/"
        RENDER_DIR="$ROOT_DIR/rendered"
        rm -rf "$RENDER_DIR"
        process_templates "$ROOT_DIR/k8s" "$RENDER_DIR/k8s"
        process_templates "$ROOT_DIR/observability" "$RENDER_DIR/observability"
        log_info "Templates rendered to $RENDER_DIR"
        ;;

    *)
        echo "Usage: $0 {k8s|observability|all|render} [--dry-run]"
        echo ""
        echo "Commands:"
        echo "  k8s           Deploy Kubernetes manifests"
        echo "  observability Deploy Prometheus, Grafana, Alertmanager"
        echo "  all           Deploy everything"
        echo "  render        Render templates to ./rendered/ without applying"
        echo ""
        echo "Options:"
        echo "  --dry-run     Validate without applying"
        exit 1
        ;;
esac

log_info "Done!"
