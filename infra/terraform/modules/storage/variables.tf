################################################################################
# Storage Module Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the storage buckets"
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
# Models Bucket Configuration
################################################################################

variable "models_bucket_name" {
  description = "Base name for the ML models bucket"
  type        = string
  default     = "sentinel-models"
}

variable "models_bucket_storage_class" {
  description = "Storage class for the models bucket"
  type        = string
  default     = "STANDARD"
  validation {
    condition     = contains(["STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"], var.models_bucket_storage_class)
    error_message = "Storage class must be one of: STANDARD, NEARLINE, COLDLINE, ARCHIVE."
  }
}

variable "models_versioning_enabled" {
  description = "Enable versioning for models bucket"
  type        = bool
  default     = true
}

variable "models_retention_days" {
  description = "Number of days to retain model versions"
  type        = number
  default     = 90
}

################################################################################
# Logs Bucket Configuration
################################################################################

variable "logs_bucket_name" {
  description = "Base name for the logs bucket"
  type        = string
  default     = "sentinel-logs"
}

variable "logs_bucket_storage_class" {
  description = "Storage class for the logs bucket"
  type        = string
  default     = "STANDARD"
}

variable "logs_versioning_enabled" {
  description = "Enable versioning for logs bucket"
  type        = bool
  default     = false
}

variable "logs_retention_days" {
  description = "Number of days to retain logs before transitioning to NEARLINE"
  type        = number
  default     = 30
}

variable "logs_archive_days" {
  description = "Number of days before archiving logs (COLDLINE)"
  type        = number
  default     = 90
}

variable "logs_delete_days" {
  description = "Number of days before deleting logs"
  type        = number
  default     = 365
}

################################################################################
# Data Bucket Configuration
################################################################################

variable "enable_data_bucket" {
  description = "Enable bucket for training data"
  type        = bool
  default     = true
}

variable "data_bucket_name" {
  description = "Base name for the training data bucket"
  type        = string
  default     = "sentinel-data"
}

variable "data_bucket_storage_class" {
  description = "Storage class for the data bucket"
  type        = string
  default     = "STANDARD"
}

variable "data_versioning_enabled" {
  description = "Enable versioning for data bucket"
  type        = bool
  default     = true
}

################################################################################
# Security Configuration
################################################################################

variable "uniform_bucket_level_access" {
  description = "Enable uniform bucket-level access"
  type        = bool
  default     = true
}

variable "public_access_prevention" {
  description = "Public access prevention setting"
  type        = string
  default     = "enforced"
  validation {
    condition     = contains(["enforced", "inherited"], var.public_access_prevention)
    error_message = "Public access prevention must be 'enforced' or 'inherited'."
  }
}

variable "enable_cmek" {
  description = "Enable Customer Managed Encryption Keys"
  type        = bool
  default     = false
}

variable "kms_key_name" {
  description = "KMS key name for CMEK encryption"
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
