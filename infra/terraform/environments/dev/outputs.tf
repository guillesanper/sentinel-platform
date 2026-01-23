################################################################################
# Dev Environment Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

################################################################################
# VPC Outputs
################################################################################

output "vpc_name" {
  description = "Name of the VPC"
  value       = module.vpc.vpc_name
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "subnet_name" {
  description = "Name of the GKE subnet"
  value       = module.vpc.subnet_name
}

################################################################################
# GKE Outputs
################################################################################

output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = module.gke.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "CA certificate of the GKE cluster"
  value       = module.gke.cluster_ca_certificate
  sensitive   = true
}

output "get_credentials_command" {
  description = "Command to get GKE credentials"
  value       = module.gke.get_credentials_command
}

################################################################################
# Storage Outputs
################################################################################

output "models_bucket_name" {
  description = "Name of the models bucket"
  value       = module.storage.models_bucket_name
}

output "logs_bucket_name" {
  description = "Name of the logs bucket"
  value       = module.storage.logs_bucket_name
}

output "data_bucket_name" {
  description = "Name of the data bucket"
  value       = module.storage.data_bucket_name
}

output "bucket_paths" {
  description = "GCS paths for storage"
  value       = module.storage.bucket_paths
}

################################################################################
# IAM Outputs
################################################################################

output "api_service_account_email" {
  description = "Email of the API service account"
  value       = module.iam.api_service_account_email
}

output "ml_service_account_email" {
  description = "Email of the ML service account"
  value       = module.iam.ml_service_account_email
}

output "workload_identity_pool" {
  description = "Workload Identity pool"
  value       = module.iam.workload_identity_pool
}

output "kubernetes_service_accounts" {
  description = "Kubernetes service account configuration"
  value       = module.iam.kubernetes_service_accounts
}

################################################################################
# Security Outputs
################################################################################

output "secret_names" {
  description = "Created secret names"
  value       = module.security.secret_names
}

################################################################################
# Quick Start Commands
################################################################################

output "quick_start" {
  description = "Quick start commands for development"
  value       = <<-EOT

    # 1. Get GKE credentials
    ${module.gke.get_credentials_command}

    # 2. Verify cluster access
    kubectl get nodes

    # 3. Create namespaces
    kubectl apply -f ../../k8s/base/namespaces/

    # 4. Set secret values (replace with actual values)
    echo -n "your-jwt-secret" | gcloud secrets versions add jwt-secret-dev --data-file=- --project=${var.project_id}
    echo -n "your-api-key" | gcloud secrets versions add api-key-dev --data-file=- --project=${var.project_id}

    # 5. Upload ML models
    gsutil cp models/*.joblib gs://${module.storage.models_bucket_name}/

    # 6. Deploy application
    kubectl apply -k ../../k8s/overlays/dev/

  EOT
}
