#!/bin/bash
set -euo pipefail

# Sentinel Platform - Kubernetes Deployment Script
#
# This script applies Kustomize overlays to deploy the application.
# Secrets are managed by External Secrets Operator (ESO) which syncs
# from GCP Secret Manager automatically.
#
# Prerequisites:
# 1. Terraform applied (infra/terraform/environments/{env}/)
# 2. kubectl configured with cluster credentials
# 3. PROJECT_ID replaced in k8s/overlays/{env}/kustomization.yaml
#
# Usage:
#   ./deploy-k8s.sh [environment] [--dry-run]
#
# Examples:
#   ./deploy-k8s.sh dev              # Deploy to dev
#   ./deploy-k8s.sh prod --dry-run   # Validate prod manifests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
ENVIRONMENT="${1:-dev}"
DRY_RUN=""

if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run=client"
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    echo "Usage: $0 {dev|staging|prod} [--dry-run]"
    exit 1
fi

OVERLAY_DIR="$ROOT_DIR/k8s/overlays/$ENVIRONMENT"

if [[ ! -d "$OVERLAY_DIR" ]]; then
    log_error "Overlay not found: $OVERLAY_DIR"
    exit 1
fi

# Check for PROJECT_ID placeholder
if grep -q "PROJECT_ID" "$OVERLAY_DIR/kustomization.yaml"; then
    log_warn "PROJECT_ID placeholder found in $OVERLAY_DIR/kustomization.yaml"
    log_warn "Please replace 'PROJECT_ID' with your actual GCP project ID"
    echo ""
    echo "You can use sed to replace it:"
    echo "  sed -i 's/PROJECT_ID/your-actual-project-id/g' $OVERLAY_DIR/kustomization.yaml"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log_info "Deploying to environment: $ENVIRONMENT"
log_info "Overlay directory: $OVERLAY_DIR"

if [[ -n "$DRY_RUN" ]]; then
    log_info "Running in dry-run mode (validation only)"
fi

# Apply Kustomize
log_info "Applying Kustomize manifests..."
kubectl apply -k "$OVERLAY_DIR" $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/sentinel-api -n platform --timeout=300s || true

    log_info "Checking External Secrets sync status..."
    kubectl get externalsecrets -n platform 2>/dev/null || log_warn "External Secrets not found - ensure Terraform has been applied"
fi

log_info "Done!"

# Show useful commands
echo ""
log_info "Useful commands:"
echo "  kubectl get pods -n platform"
echo "  kubectl get externalsecrets -n platform"
echo "  kubectl logs -f deployment/sentinel-api -n platform"
echo "  kubectl port-forward svc/sentinel-api 8000:80 -n platform"
