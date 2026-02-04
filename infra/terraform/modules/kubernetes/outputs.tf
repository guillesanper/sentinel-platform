################################################################################
# Kubernetes Module - Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

output "eso_namespace" {
  description = "Namespace where External Secrets Operator is installed"
  value       = var.install_eso ? kubernetes_namespace.eso[0].metadata[0].name : null
}

output "cluster_secret_store_name" {
  description = "Name of the ClusterSecretStore for GCP Secret Manager"
  value       = var.install_eso ? kubectl_manifest.cluster_secret_store[0].name : null
}

output "synced_secrets" {
  description = "List of secrets being synced from GCP to Kubernetes"
  value       = var.secrets_to_sync
}

output "target_namespaces" {
  description = "Namespaces where secrets are synced"
  value       = var.target_namespaces
}

output "eso_service_account" {
  description = "Kubernetes service account used by ESO"
  value       = var.install_eso ? "external-secrets" : null
}

# Quick reference for kubectl commands
output "verify_commands" {
  description = "Commands to verify ESO installation"
  value = var.install_eso ? {
    check_eso_pods     = "kubectl get pods -n ${var.eso_namespace}"
    check_secret_store = "kubectl get clustersecretstore"
    check_secrets      = "kubectl get externalsecrets -A"
  } : null
}
