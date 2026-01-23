################################################################################
# Storage Module Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

################################################################################
# Models Bucket Outputs
################################################################################

output "models_bucket_name" {
  description = "Name of the ML models bucket"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "URL of the ML models bucket"
  value       = google_storage_bucket.models.url
}

output "models_bucket_self_link" {
  description = "Self link of the ML models bucket"
  value       = google_storage_bucket.models.self_link
}

################################################################################
# Logs Bucket Outputs
################################################################################

output "logs_bucket_name" {
  description = "Name of the logs bucket"
  value       = google_storage_bucket.logs.name
}

output "logs_bucket_url" {
  description = "URL of the logs bucket"
  value       = google_storage_bucket.logs.url
}

output "logs_bucket_self_link" {
  description = "Self link of the logs bucket"
  value       = google_storage_bucket.logs.self_link
}

################################################################################
# Data Bucket Outputs
################################################################################

output "data_bucket_name" {
  description = "Name of the training data bucket"
  value       = var.enable_data_bucket ? google_storage_bucket.data[0].name : null
}

output "data_bucket_url" {
  description = "URL of the training data bucket"
  value       = var.enable_data_bucket ? google_storage_bucket.data[0].url : null
}

output "data_bucket_self_link" {
  description = "Self link of the training data bucket"
  value       = var.enable_data_bucket ? google_storage_bucket.data[0].self_link : null
}

################################################################################
# All Buckets
################################################################################

output "all_bucket_names" {
  description = "List of all bucket names"
  value = compact([
    google_storage_bucket.models.name,
    google_storage_bucket.logs.name,
    var.enable_data_bucket ? google_storage_bucket.data[0].name : "",
  ])
}

output "bucket_paths" {
  description = "GCS paths for different storage types"
  value = {
    models = {
      anomaly_detector = "gs://${google_storage_bucket.models.name}/anomaly_detector/"
      classifier       = "gs://${google_storage_bucket.models.name}/classifier/"
      archive          = "gs://${google_storage_bucket.models.name}/archive/"
    }
    logs = {
      api         = "gs://${google_storage_bucket.logs.name}/api/"
      predictions = "gs://${google_storage_bucket.logs.name}/predictions/"
      audit       = "gs://${google_storage_bucket.logs.name}/audit/"
    }
    data = var.enable_data_bucket ? {
      raw       = "gs://${google_storage_bucket.data[0].name}/raw/"
      processed = "gs://${google_storage_bucket.data[0].name}/processed/"
      features  = "gs://${google_storage_bucket.data[0].name}/features/"
    } : null
  }
}
