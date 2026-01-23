################################################################################
# Storage Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Creates GCS buckets for:
# - ML models storage with versioning
# - Logs storage with lifecycle policies
# - Training data storage (optional)
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
  models_bucket_name = "${var.models_bucket_name}-${var.environment}-${var.project_id}"
  logs_bucket_name   = "${var.logs_bucket_name}-${var.environment}-${var.project_id}"
  data_bucket_name   = "${var.data_bucket_name}-${var.environment}-${var.project_id}"

  default_labels = {
    project     = "sentinel"
    environment = var.environment
    managed_by  = "terraform"
  }
  labels = merge(local.default_labels, var.labels)
}

################################################################################
# ML Models Bucket
################################################################################

resource "google_storage_bucket" "models" {
  name          = local.models_bucket_name
  project       = var.project_id
  location      = var.region
  storage_class = var.models_bucket_storage_class

  # Security settings
  uniform_bucket_level_access = var.uniform_bucket_level_access
  public_access_prevention    = var.public_access_prevention

  # Versioning for model rollback capability
  versioning {
    enabled = var.models_versioning_enabled
  }

  # Lifecycle rules for model versions
  lifecycle_rule {
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age        = var.models_retention_days
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  # CMEK encryption
  dynamic "encryption" {
    for_each = var.enable_cmek && var.kms_key_name != "" ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }

  # Soft delete policy (30 days)
  soft_delete_policy {
    retention_duration_seconds = 2592000 # 30 days
  }

  labels = merge(local.labels, {
    bucket_type = "models"
  })

  # Force destroy only in dev
  force_destroy = var.environment == "dev" ? true : false
}

################################################################################
# Logs Bucket
################################################################################

resource "google_storage_bucket" "logs" {
  name          = local.logs_bucket_name
  project       = var.project_id
  location      = var.region
  storage_class = var.logs_bucket_storage_class

  # Security settings
  uniform_bucket_level_access = var.uniform_bucket_level_access
  public_access_prevention    = var.public_access_prevention

  # Versioning
  versioning {
    enabled = var.logs_versioning_enabled
  }

  # Lifecycle rules for cost optimization
  lifecycle_rule {
    condition {
      age = var.logs_retention_days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = var.logs_archive_days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = var.logs_delete_days
    }
    action {
      type = "Delete"
    }
  }

  # CMEK encryption
  dynamic "encryption" {
    for_each = var.enable_cmek && var.kms_key_name != "" ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }

  # Soft delete policy (7 days for logs)
  soft_delete_policy {
    retention_duration_seconds = 604800 # 7 days
  }

  labels = merge(local.labels, {
    bucket_type = "logs"
  })

  # Force destroy only in dev
  force_destroy = var.environment == "dev" ? true : false
}

################################################################################
# Training Data Bucket (Optional)
################################################################################

resource "google_storage_bucket" "data" {
  count = var.enable_data_bucket ? 1 : 0

  name          = local.data_bucket_name
  project       = var.project_id
  location      = var.region
  storage_class = var.data_bucket_storage_class

  # Security settings
  uniform_bucket_level_access = var.uniform_bucket_level_access
  public_access_prevention    = var.public_access_prevention

  # Versioning for data integrity
  versioning {
    enabled = var.data_versioning_enabled
  }

  # Keep latest 3 versions
  lifecycle_rule {
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  # CMEK encryption
  dynamic "encryption" {
    for_each = var.enable_cmek && var.kms_key_name != "" ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }

  # Soft delete policy
  soft_delete_policy {
    retention_duration_seconds = 2592000 # 30 days
  }

  labels = merge(local.labels, {
    bucket_type = "data"
  })

  # Force destroy only in dev
  force_destroy = var.environment == "dev" ? true : false
}

################################################################################
# Bucket Folders (for organization)
################################################################################

# Models bucket folders
resource "google_storage_bucket_object" "models_anomaly_detector" {
  name    = "anomaly_detector/.gitkeep"
  content = "# Anomaly detector models"
  bucket  = google_storage_bucket.models.name
}

resource "google_storage_bucket_object" "models_classifier" {
  name    = "classifier/.gitkeep"
  content = "# Attack classifier models"
  bucket  = google_storage_bucket.models.name
}

resource "google_storage_bucket_object" "models_archive" {
  name    = "archive/.gitkeep"
  content = "# Archived model versions"
  bucket  = google_storage_bucket.models.name
}

# Logs bucket folders
resource "google_storage_bucket_object" "logs_api" {
  name    = "api/.gitkeep"
  content = "# API request logs"
  bucket  = google_storage_bucket.logs.name
}

resource "google_storage_bucket_object" "logs_predictions" {
  name    = "predictions/.gitkeep"
  content = "# Prediction results logs"
  bucket  = google_storage_bucket.logs.name
}

resource "google_storage_bucket_object" "logs_audit" {
  name    = "audit/.gitkeep"
  content = "# Audit logs"
  bucket  = google_storage_bucket.logs.name
}

# Data bucket folders (if enabled)
resource "google_storage_bucket_object" "data_raw" {
  count = var.enable_data_bucket ? 1 : 0

  name    = "raw/.gitkeep"
  content = "# Raw training data"
  bucket  = google_storage_bucket.data[0].name
}

resource "google_storage_bucket_object" "data_processed" {
  count = var.enable_data_bucket ? 1 : 0

  name    = "processed/.gitkeep"
  content = "# Processed training data"
  bucket  = google_storage_bucket.data[0].name
}

resource "google_storage_bucket_object" "data_features" {
  count = var.enable_data_bucket ? 1 : 0

  name    = "features/.gitkeep"
  content = "# Extracted features"
  bucket  = google_storage_bucket.data[0].name
}
