################################################################################
# Terraform Backend Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# This file configures the GCS backend for Terraform state storage.
# State is stored remotely for team collaboration and state locking.
################################################################################

# Backend configuration is defined in each environment's main.tf
# This file provides the bootstrap configuration for creating the state bucket.

# To bootstrap the infrastructure:
#
# 1. First, create the GCS bucket for state storage:
#
#    gcloud storage buckets create gs://sentinel-terraform-state-${PROJECT_ID} \
#      --project=${PROJECT_ID} \
#      --location=US \
#      --uniform-bucket-level-access \
#      --public-access-prevention
#
# 2. Enable versioning for state recovery:
#
#    gcloud storage buckets update gs://sentinel-terraform-state-${PROJECT_ID} \
#      --versioning
#
# 3. Enable the required APIs:
#
#    gcloud services enable \
#      compute.googleapis.com \
#      container.googleapis.com \
#      storage.googleapis.com \
#      secretmanager.googleapis.com \
#      cloudkms.googleapis.com \
#      servicenetworking.googleapis.com \
#      cloudresourcemanager.googleapis.com \
#      iam.googleapis.com \
#      --project=${PROJECT_ID}
#
# 4. Then initialize Terraform in the environment directory:
#
#    cd environments/dev
#    terraform init
#
# Note: The backend block in each environment's main.tf should be uncommented
# and configured with the appropriate bucket and prefix.

################################################################################
# State Bucket Module (for bootstrapping)
################################################################################

# Uncomment this module to create the state bucket using Terraform itself.
# Run with local state first, then migrate to remote state.

# terraform {
#   required_version = ">= 1.5.0"
#   required_providers {
#     google = {
#       source  = "hashicorp/google"
#       version = "~> 5.0"
#     }
#   }
# }
#
# variable "project_id" {
#   description = "GCP project ID"
#   type        = string
# }
#
# variable "region" {
#   description = "GCP region for the state bucket"
#   type        = string
#   default     = "US"
# }
#
# resource "google_storage_bucket" "terraform_state" {
#   name          = "sentinel-terraform-state-${var.project_id}"
#   project       = var.project_id
#   location      = var.region
#   storage_class = "STANDARD"
#
#   uniform_bucket_level_access = true
#   public_access_prevention    = "enforced"
#
#   versioning {
#     enabled = true
#   }
#
#   lifecycle_rule {
#     condition {
#       num_newer_versions = 10
#     }
#     action {
#       type = "Delete"
#     }
#   }
#
#   labels = {
#     project    = "sentinel"
#     purpose    = "terraform-state"
#     managed_by = "terraform"
#   }
# }
#
# output "state_bucket_name" {
#   description = "Name of the Terraform state bucket"
#   value       = google_storage_bucket.terraform_state.name
# }
