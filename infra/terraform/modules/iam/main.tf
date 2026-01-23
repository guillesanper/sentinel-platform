################################################################################
# IAM Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Creates service accounts and IAM bindings:
# - sentinel-api: For API pods (read models, write logs)
# - sentinel-ml: For ML training jobs (read/write models, read data)
# - Workload Identity bindings for GKE
################################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

locals {
  api_sa_name = "sentinel-api-${var.environment}"
  ml_sa_name  = "sentinel-ml-${var.environment}"

  # Workload Identity pool
  workload_identity_pool = "${var.project_id}.svc.id.goog"

  # Workload Identity members
  api_wi_member = "serviceAccount:${local.workload_identity_pool}[${var.api_k8s_namespace}/${var.api_k8s_service_account}]"
  ml_wi_member  = "serviceAccount:${local.workload_identity_pool}[${var.ml_k8s_namespace}/${var.ml_k8s_service_account}]"
}

################################################################################
# Custom IAM Roles (Least Privilege)
################################################################################

# Custom role for API service
resource "google_project_iam_custom_role" "api_role" {
  count = var.create_custom_roles ? 1 : 0

  role_id     = "sentinelApi${title(var.environment)}"
  project     = var.project_id
  title       = "Sentinel API ${title(var.environment)}"
  description = "Custom role for Sentinel API service with least privilege"

  permissions = [
    # Storage - Read models
    "storage.objects.get",
    "storage.objects.list",
    # Storage - Write logs
    "storage.objects.create",
    # Logging
    "logging.logEntries.create",
    # Monitoring
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.metricDescriptors.list",
    "monitoring.timeSeries.create",
    # Secret Manager - Read
    "secretmanager.versions.access",
  ]
}

# Custom role for ML workloads
resource "google_project_iam_custom_role" "ml_role" {
  count = var.create_custom_roles ? 1 : 0

  role_id     = "sentinelMl${title(var.environment)}"
  project     = var.project_id
  title       = "Sentinel ML ${title(var.environment)}"
  description = "Custom role for Sentinel ML workloads with least privilege"

  permissions = [
    # Storage - Full access to models and data
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.update",
    # Logging
    "logging.logEntries.create",
    # Monitoring
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.metricDescriptors.list",
    "monitoring.timeSeries.create",
    # Secret Manager - Read
    "secretmanager.versions.access",
    # AI Platform (for MLflow integration)
    "aiplatform.endpoints.predict",
  ]
}

################################################################################
# Service Account for API
################################################################################

resource "google_service_account" "api" {
  account_id   = local.api_sa_name
  project      = var.project_id
  display_name = "Sentinel API Service Account (${var.environment})"
  description  = "Service account for Sentinel API pods"
}

# Grant custom role to API SA
resource "google_project_iam_member" "api_custom_role" {
  count = var.create_custom_roles ? 1 : 0

  project = var.project_id
  role    = google_project_iam_custom_role.api_role[0].id
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Alternative: Use predefined roles if custom roles are disabled
resource "google_project_iam_member" "api_storage_viewer" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_log_writer" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_monitoring_writer" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.api.email}"
}

################################################################################
# Service Account for ML Workloads
################################################################################

resource "google_service_account" "ml" {
  account_id   = local.ml_sa_name
  project      = var.project_id
  display_name = "Sentinel ML Service Account (${var.environment})"
  description  = "Service account for Sentinel ML training jobs"
}

# Grant custom role to ML SA
resource "google_project_iam_member" "ml_custom_role" {
  count = var.create_custom_roles ? 1 : 0

  project = var.project_id
  role    = google_project_iam_custom_role.ml_role[0].id
  member  = "serviceAccount:${google_service_account.ml.email}"
}

# Alternative: Use predefined roles if custom roles are disabled
resource "google_project_iam_member" "ml_storage_admin" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

resource "google_project_iam_member" "ml_log_writer" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

resource "google_project_iam_member" "ml_monitoring_writer" {
  count = var.create_custom_roles ? 0 : 1

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

################################################################################
# Bucket-Level IAM Bindings
################################################################################

# API SA - Read access to models bucket
resource "google_storage_bucket_iam_member" "api_models_reader" {
  bucket = var.models_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# API SA - Write access to logs bucket
resource "google_storage_bucket_iam_member" "api_logs_writer" {
  bucket = var.logs_bucket_name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.api.email}"
}

# ML SA - Admin access to models bucket
resource "google_storage_bucket_iam_member" "ml_models_admin" {
  bucket = var.models_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ml.email}"
}

# ML SA - Read access to data bucket
resource "google_storage_bucket_iam_member" "ml_data_reader" {
  count = var.data_bucket_name != "" ? 1 : 0

  bucket = var.data_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ml.email}"
}

# ML SA - Write access to logs bucket
resource "google_storage_bucket_iam_member" "ml_logs_writer" {
  bucket = var.logs_bucket_name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.ml.email}"
}

################################################################################
# Secret Manager IAM Bindings
################################################################################

# API SA - Access to secrets
resource "google_secret_manager_secret_iam_member" "api_secrets" {
  for_each = toset(var.secret_ids)

  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# ML SA - Access to secrets
resource "google_secret_manager_secret_iam_member" "ml_secrets" {
  for_each = toset(var.secret_ids)

  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ml.email}"
}

################################################################################
# Workload Identity Bindings
################################################################################

# Allow GKE to impersonate API SA
resource "google_service_account_iam_member" "api_workload_identity" {
  count = var.enable_workload_identity ? 1 : 0

  service_account_id = google_service_account.api.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.api_wi_member
}

# Allow GKE to impersonate ML SA
resource "google_service_account_iam_member" "ml_workload_identity" {
  count = var.enable_workload_identity ? 1 : 0

  service_account_id = google_service_account.ml.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.ml_wi_member
}
