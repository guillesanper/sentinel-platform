################################################################################
# Kubernetes Module - Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name for authentication"
  type        = string
}

variable "cluster_endpoint" {
  description = "GKE cluster endpoint"
  type        = string
  sensitive   = true
}

variable "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (base64 encoded)"
  type        = string
  sensitive   = true
}

# External Secrets Operator configuration
variable "install_eso" {
  description = "Whether to install External Secrets Operator"
  type        = bool
  default     = true
}

variable "eso_namespace" {
  description = "Namespace for External Secrets Operator"
  type        = string
  default     = "external-secrets"
}

variable "eso_version" {
  description = "External Secrets Operator Helm chart version"
  type        = string
  default     = "0.9.11"
}

# Secrets configuration
variable "secrets_to_sync" {
  description = "List of GCP Secret Manager secrets to sync to Kubernetes"
  type        = list(string)
  default     = ["jwt-secret", "api-key"]
}

variable "target_namespaces" {
  description = "Namespaces where secrets should be synced"
  type        = list(string)
  default     = ["platform", "ml"]
}

# Service account for ESO
variable "eso_service_account_email" {
  description = "GCP Service Account email for ESO to access Secret Manager"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
