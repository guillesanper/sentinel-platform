################################################################################
# VPC Module Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the VPC"
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

variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "sentinel-vpc"
}

variable "nodes_cidr" {
  description = "CIDR range for GKE nodes subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "CIDR range for GKE pods (secondary range)"
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_cidr" {
  description = "CIDR range for GKE services (secondary range)"
  type        = string
  default     = "10.8.0.0/20"
}

variable "master_cidr" {
  description = "CIDR range for GKE master nodes (private cluster)"
  type        = string
  default     = "172.16.0.0/28"
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs for network monitoring"
  type        = bool
  default     = true
}

variable "flow_logs_interval" {
  description = "Aggregation interval for flow logs"
  type        = string
  default     = "INTERVAL_5_SEC"
  validation {
    condition     = contains(["INTERVAL_5_SEC", "INTERVAL_30_SEC", "INTERVAL_1_MIN", "INTERVAL_5_MIN", "INTERVAL_10_MIN", "INTERVAL_15_MIN"], var.flow_logs_interval)
    error_message = "Invalid flow logs interval."
  }
}

variable "flow_logs_sampling" {
  description = "Sampling rate for flow logs (0.0 to 1.0)"
  type        = number
  default     = 0.5
  validation {
    condition     = var.flow_logs_sampling >= 0 && var.flow_logs_sampling <= 1
    error_message = "Flow logs sampling must be between 0.0 and 1.0."
  }
}

variable "allowed_master_cidr_blocks" {
  description = "CIDR blocks allowed to access the GKE master endpoint"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}
