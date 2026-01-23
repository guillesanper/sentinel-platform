################################################################################
# IAM Module Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

################################################################################
# Workload Identity Configuration
################################################################################

variable "enable_workload_identity" {
  description = "Enable Workload Identity bindings"
  type        = bool
  default     = true
}

variable "gke_cluster_name" {
  description = "Name of the GKE cluster for Workload Identity"
  type        = string
  default     = ""
}

variable "api_k8s_namespace" {
  description = "Kubernetes namespace for the API service"
  type        = string
  default     = "platform"
}

variable "api_k8s_service_account" {
  description = "Kubernetes service account name for the API"
  type        = string
  default     = "sentinel-api"
}

variable "ml_k8s_namespace" {
  description = "Kubernetes namespace for ML workloads"
  type        = string
  default     = "ml"
}

variable "ml_k8s_service_account" {
  description = "Kubernetes service account name for ML workloads"
  type        = string
  default     = "sentinel-ml"
}

################################################################################
# Storage Buckets (for IAM bindings)
################################################################################

variable "models_bucket_name" {
  description = "Name of the ML models bucket"
  type        = string
}

variable "logs_bucket_name" {
  description = "Name of the logs bucket"
  type        = string
}

variable "data_bucket_name" {
  description = "Name of the training data bucket (optional)"
  type        = string
  default     = ""
}

################################################################################
# Secret Manager Configuration
################################################################################

variable "secret_ids" {
  description = "List of Secret Manager secret IDs to grant access to"
  type        = list(string)
  default     = []
}

################################################################################
# Custom IAM Roles
################################################################################

variable "create_custom_roles" {
  description = "Create custom IAM roles with least privilege"
  type        = bool
  default     = true
}

################################################################################
# Labels
################################################################################

variable "labels" {
  description = "Labels to apply to service accounts"
  type        = map(string)
  default     = {}
}
