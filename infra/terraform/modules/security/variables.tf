################################################################################
# Security Module Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
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
# Cloud Armor Configuration
################################################################################

variable "enable_cloud_armor" {
  description = "Enable Cloud Armor WAF"
  type        = bool
  default     = true
}

variable "cloud_armor_policy_name" {
  description = "Name of the Cloud Armor security policy"
  type        = string
  default     = "sentinel-security-policy"
}

variable "allowed_source_ips" {
  description = "List of IP ranges allowed to access the API"
  type        = list(string)
  default     = []
}

variable "rate_limit_threshold" {
  description = "Rate limit threshold (requests per interval)"
  type        = number
  default     = 1000
}

variable "rate_limit_interval_sec" {
  description = "Rate limit interval in seconds"
  type        = number
  default     = 60
}

variable "enable_owasp_rules" {
  description = "Enable OWASP ModSecurity Core Rule Set"
  type        = bool
  default     = true
}

variable "owasp_sensitivity_level" {
  description = "OWASP rules sensitivity (1-4, higher = more sensitive)"
  type        = number
  default     = 2
  validation {
    condition     = var.owasp_sensitivity_level >= 1 && var.owasp_sensitivity_level <= 4
    error_message = "OWASP sensitivity level must be between 1 and 4."
  }
}

################################################################################
# Secret Manager Configuration
################################################################################

variable "create_secrets" {
  description = "Create Secret Manager secrets"
  type        = bool
  default     = true
}

variable "secrets" {
  description = "Map of secrets to create"
  type = map(object({
    description = string
    labels      = optional(map(string), {})
  }))
  default = {
    jwt-secret = {
      description = "JWT signing secret for API authentication"
    }
    api-key = {
      description = "API key for service-to-service authentication"
    }
    mlflow-tracking-uri = {
      description = "MLflow tracking server URI"
    }
  }
}

variable "secret_replication_locations" {
  description = "Locations for secret replication"
  type        = list(string)
  default     = []
}

################################################################################
# KMS Configuration
################################################################################

variable "enable_kms" {
  description = "Enable Cloud KMS for encryption"
  type        = bool
  default     = false
}

variable "kms_keyring_name" {
  description = "Name of the KMS keyring"
  type        = string
  default     = "sentinel-keyring"
}

variable "kms_key_rotation_period" {
  description = "Key rotation period in seconds"
  type        = string
  default     = "7776000s" # 90 days
}

################################################################################
# VPC Service Controls
################################################################################

variable "enable_vpc_service_controls" {
  description = "Enable VPC Service Controls"
  type        = bool
  default     = false
}

variable "access_policy_name" {
  description = "Name of the access policy for VPC-SC"
  type        = string
  default     = ""
}

################################################################################
# Labels
################################################################################

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}
