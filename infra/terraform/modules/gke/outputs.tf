################################################################################
# GKE Module Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

output "cluster_id" {
  description = "The ID of the GKE cluster"
  value       = google_container_cluster.cluster.id
}

output "cluster_name" {
  description = "The name of the GKE cluster"
  value       = google_container_cluster.cluster.name
}

output "cluster_self_link" {
  description = "The self link of the GKE cluster"
  value       = google_container_cluster.cluster.self_link
}

output "cluster_endpoint" {
  description = "The endpoint for the GKE cluster master"
  value       = google_container_cluster.cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "The CA certificate for the GKE cluster"
  value       = google_container_cluster.cluster.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "The location of the GKE cluster"
  value       = google_container_cluster.cluster.location
}

output "cluster_master_version" {
  description = "The current master version of the GKE cluster"
  value       = google_container_cluster.cluster.master_version
}

output "workload_identity_pool" {
  description = "The Workload Identity pool for the cluster"
  value       = var.enable_workload_identity ? "${var.project_id}.svc.id.goog" : null
}

output "default_node_pool_name" {
  description = "The name of the default node pool"
  value       = google_container_node_pool.default.name
}

output "ml_node_pool_name" {
  description = "The name of the ML node pool"
  value       = var.enable_ml_node_pool ? google_container_node_pool.ml[0].name : null
}

# Connection command for kubectl
output "get_credentials_command" {
  description = "gcloud command to get cluster credentials"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.cluster.name} --region ${var.region} --project ${var.project_id}"
}

# For use with Kubernetes provider
output "kubernetes_provider_config" {
  description = "Configuration for Kubernetes provider"
  value = {
    host                   = "https://${google_container_cluster.cluster.endpoint}"
    cluster_ca_certificate = base64decode(google_container_cluster.cluster.master_auth[0].cluster_ca_certificate)
  }
  sensitive = true
}
