################################################################################
# Dev Environment Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for single-zone resources"
  type        = string
  default     = "us-central1-a"
}
