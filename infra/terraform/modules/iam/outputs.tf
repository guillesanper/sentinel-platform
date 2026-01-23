################################################################################
# IAM Module Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

################################################################################
# API Service Account Outputs
################################################################################

output "api_service_account_email" {
  description = "Email of the API service account"
  value       = google_service_account.api.email
}

output "api_service_account_name" {
  description = "Name of the API service account"
  value       = google_service_account.api.name
}

output "api_service_account_id" {
  description = "ID of the API service account"
  value       = google_service_account.api.id
}

output "api_service_account_unique_id" {
  description = "Unique ID of the API service account"
  value       = google_service_account.api.unique_id
}

################################################################################
# ML Service Account Outputs
################################################################################

output "ml_service_account_email" {
  description = "Email of the ML service account"
  value       = google_service_account.ml.email
}

output "ml_service_account_name" {
  description = "Name of the ML service account"
  value       = google_service_account.ml.name
}

output "ml_service_account_id" {
  description = "ID of the ML service account"
  value       = google_service_account.ml.id
}

output "ml_service_account_unique_id" {
  description = "Unique ID of the ML service account"
  value       = google_service_account.ml.unique_id
}

################################################################################
# Workload Identity Outputs
################################################################################

output "api_workload_identity_annotation" {
  description = "Annotation to add to K8s service account for API Workload Identity"
  value       = var.enable_workload_identity ? "iam.gke.io/gcp-service-account=${google_service_account.api.email}" : null
}

output "ml_workload_identity_annotation" {
  description = "Annotation to add to K8s service account for ML Workload Identity"
  value       = var.enable_workload_identity ? "iam.gke.io/gcp-service-account=${google_service_account.ml.email}" : null
}

output "workload_identity_pool" {
  description = "Workload Identity pool for the project"
  value       = var.enable_workload_identity ? "${var.project_id}.svc.id.goog" : null
}

################################################################################
# Custom Roles Outputs
################################################################################

output "api_custom_role_id" {
  description = "ID of the custom API role"
  value       = var.create_custom_roles ? google_project_iam_custom_role.api_role[0].id : null
}

output "ml_custom_role_id" {
  description = "ID of the custom ML role"
  value       = var.create_custom_roles ? google_project_iam_custom_role.ml_role[0].id : null
}

################################################################################
# K8s Service Account Configuration
################################################################################

output "kubernetes_service_accounts" {
  description = "Kubernetes service account configuration for Workload Identity"
  value = {
    api = {
      namespace       = var.api_k8s_namespace
      name            = var.api_k8s_service_account
      gcp_sa_email    = google_service_account.api.email
      annotation_key  = "iam.gke.io/gcp-service-account"
      annotation_value = google_service_account.api.email
    }
    ml = {
      namespace       = var.ml_k8s_namespace
      name            = var.ml_k8s_service_account
      gcp_sa_email    = google_service_account.ml.email
      annotation_key  = "iam.gke.io/gcp-service-account"
      annotation_value = google_service_account.ml.email
    }
  }
}
