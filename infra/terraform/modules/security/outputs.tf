################################################################################
# Security Module Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

################################################################################
# Cloud Armor Outputs
################################################################################

output "security_policy_id" {
  description = "ID of the Cloud Armor security policy"
  value       = var.enable_cloud_armor ? google_compute_security_policy.policy[0].id : null
}

output "security_policy_name" {
  description = "Name of the Cloud Armor security policy"
  value       = var.enable_cloud_armor ? google_compute_security_policy.policy[0].name : null
}

output "security_policy_self_link" {
  description = "Self link of the Cloud Armor security policy"
  value       = var.enable_cloud_armor ? google_compute_security_policy.policy[0].self_link : null
}

output "owasp_policy_id" {
  description = "ID of the OWASP Cloud Armor policy"
  value       = var.enable_cloud_armor && var.enable_owasp_rules ? google_compute_security_policy.owasp_policy[0].id : null
}

output "owasp_policy_name" {
  description = "Name of the OWASP Cloud Armor policy"
  value       = var.enable_cloud_armor && var.enable_owasp_rules ? google_compute_security_policy.owasp_policy[0].name : null
}

################################################################################
# Secret Manager Outputs
################################################################################

output "secret_ids" {
  description = "Map of secret names to their IDs"
  value = var.create_secrets ? {
    for k, v in google_secret_manager_secret.secrets : k => v.id
  } : {}
}

output "secret_names" {
  description = "Map of secret names to their full names"
  value = var.create_secrets ? {
    for k, v in google_secret_manager_secret.secrets : k => v.secret_id
  } : {}
}

output "secret_versions_to_create" {
  description = "Instructions for creating initial secret versions"
  value = var.create_secrets ? {
    for k, v in google_secret_manager_secret.secrets : k => "gcloud secrets versions add ${v.secret_id} --data-file=/path/to/${k}.txt --project=${var.project_id}"
  } : {}
}

################################################################################
# KMS Outputs
################################################################################

output "kms_keyring_id" {
  description = "ID of the KMS keyring"
  value       = var.enable_kms ? google_kms_key_ring.keyring[0].id : null
}

output "kms_keyring_name" {
  description = "Name of the KMS keyring"
  value       = var.enable_kms ? google_kms_key_ring.keyring[0].name : null
}

output "storage_kms_key_id" {
  description = "ID of the storage encryption key"
  value       = var.enable_kms ? google_kms_crypto_key.storage_key[0].id : null
}

output "storage_kms_key_name" {
  description = "Name of the storage encryption key"
  value       = var.enable_kms ? google_kms_crypto_key.storage_key[0].name : null
}

output "secrets_kms_key_id" {
  description = "ID of the secrets encryption key"
  value       = var.enable_kms ? google_kms_crypto_key.secrets_key[0].id : null
}

################################################################################
# Configuration for Other Modules
################################################################################

output "security_config" {
  description = "Security configuration for use in other modules"
  value = {
    cloud_armor_enabled     = var.enable_cloud_armor
    security_policy_name    = var.enable_cloud_armor ? google_compute_security_policy.policy[0].name : null
    kms_enabled            = var.enable_kms
    storage_kms_key        = var.enable_kms ? google_kms_crypto_key.storage_key[0].id : null
    secrets                = var.create_secrets ? keys(var.secrets) : []
  }
}
