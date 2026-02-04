################################################################################
# Kubernetes Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# This module:
# 1. Configures Kubernetes and Helm providers
# 2. Installs External Secrets Operator (ESO)
# 3. Creates ClusterSecretStore for GCP Secret Manager
# 4. Creates ExternalSecrets for each target namespace
################################################################################

################################################################################
# External Secrets Operator Installation
################################################################################

# Create namespace for ESO
resource "kubernetes_namespace" "eso" {
  count = var.install_eso ? 1 : 0

  metadata {
    name = var.eso_namespace

    labels = merge(var.labels, {
      "app.kubernetes.io/name"       = "external-secrets"
      "app.kubernetes.io/managed-by" = "terraform"
    })
  }
}

# Create GCP Service Account for ESO (Workload Identity)
resource "google_service_account" "eso" {
  count = var.install_eso && var.eso_service_account_email == "" ? 1 : 0

  account_id   = "sentinel-eso-${var.environment}"
  display_name = "External Secrets Operator SA for ${var.environment}"
  project      = var.project_id
}

# Grant Secret Manager access to ESO service account
resource "google_project_iam_member" "eso_secret_accessor" {
  count = var.install_eso && var.eso_service_account_email == "" ? 1 : 0

  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.eso[0].email}"
}

# Workload Identity binding for ESO
resource "google_service_account_iam_member" "eso_workload_identity" {
  count = var.install_eso && var.eso_service_account_email == "" ? 1 : 0

  service_account_id = google_service_account.eso[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.eso_namespace}/external-secrets]"
}

# Install External Secrets Operator via Helm
resource "helm_release" "external_secrets" {
  count = var.install_eso ? 1 : 0

  name       = "external-secrets"
  repository = "https://charts.external-secrets.io"
  chart      = "external-secrets"
  version    = var.eso_version
  namespace  = kubernetes_namespace.eso[0].metadata[0].name

  # Wait for CRDs to be ready
  wait            = true
  wait_for_jobs   = true
  cleanup_on_fail = true
  timeout         = 300

  # Configure Workload Identity
  set {
    name  = "serviceAccount.annotations.iam\\.gke\\.io/gcp-service-account"
    value = var.eso_service_account_email != "" ? var.eso_service_account_email : google_service_account.eso[0].email
  }

  set {
    name  = "serviceAccount.name"
    value = "external-secrets"
  }

  # Install CRDs
  set {
    name  = "installCRDs"
    value = "true"
  }

  depends_on = [
    kubernetes_namespace.eso,
    google_service_account_iam_member.eso_workload_identity
  ]
}

################################################################################
# ClusterSecretStore for GCP Secret Manager
################################################################################

# Wait for ESO to be ready before creating ClusterSecretStore
resource "time_sleep" "wait_for_eso" {
  count = var.install_eso ? 1 : 0

  depends_on      = [helm_release.external_secrets]
  create_duration = "30s"
}

# Create ClusterSecretStore for GCP
resource "kubectl_manifest" "cluster_secret_store" {
  count = var.install_eso ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: external-secrets.io/v1beta1
    kind: ClusterSecretStore
    metadata:
      name: gcp-secret-manager
      labels:
        app.kubernetes.io/managed-by: terraform
        environment: ${var.environment}
    spec:
      provider:
        gcpsm:
          projectID: ${var.project_id}
  YAML

  depends_on = [time_sleep.wait_for_eso]
}

################################################################################
# ExternalSecrets for each namespace
################################################################################

# Create ExternalSecrets in each target namespace
resource "kubectl_manifest" "external_secrets" {
  for_each = var.install_eso ? toset(var.target_namespaces) : []

  yaml_body = <<-YAML
    apiVersion: external-secrets.io/v1beta1
    kind: ExternalSecret
    metadata:
      name: sentinel-secrets
      namespace: ${each.value}
      labels:
        app.kubernetes.io/managed-by: terraform
        environment: ${var.environment}
    spec:
      refreshInterval: 1h
      secretStoreRef:
        kind: ClusterSecretStore
        name: gcp-secret-manager
      target:
        name: sentinel-secrets
        creationPolicy: Owner
      data:
%{for secret in var.secrets_to_sync}
        - secretKey: ${secret}
          remoteRef:
            key: ${secret}-${var.environment}
%{endfor}
  YAML

  depends_on = [kubectl_manifest.cluster_secret_store]
}

################################################################################
# Output image configuration for Kustomize
################################################################################

# Generate image configuration file for Kustomize
resource "local_file" "kustomize_image_config" {
  count = var.install_eso ? 1 : 0

  filename = "${path.root}/../../k8s/overlays/${var.environment}/image-vars.env"
  content  = <<-EOF
    # Generated by Terraform - Do not edit manually
    PROJECT_ID=${var.project_id}
    ENVIRONMENT=${var.environment}
    IMAGE_REGISTRY=gcr.io/${var.project_id}
  EOF

  file_permission = "0644"
}
