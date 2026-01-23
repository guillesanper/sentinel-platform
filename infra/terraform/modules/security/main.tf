################################################################################
# Security Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Creates security resources:
# - Cloud Armor WAF policy with OWASP rules
# - Secret Manager secrets
# - KMS keyring and keys (optional)
################################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0.0"
    }
  }
}

locals {
  policy_name = "${var.cloud_armor_policy_name}-${var.environment}"

  default_labels = {
    project     = "sentinel"
    environment = var.environment
    managed_by  = "terraform"
  }
  labels = merge(local.default_labels, var.labels)
}

################################################################################
# Cloud Armor Security Policy
################################################################################

resource "google_compute_security_policy" "policy" {
  count = var.enable_cloud_armor ? 1 : 0

  name        = local.policy_name
  project     = var.project_id
  description = "Cloud Armor security policy for Sentinel API (${var.environment})"

  # Adaptive Protection for DDoS
  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable          = true
      rule_visibility = "STANDARD"
    }
  }

  # Default rule - deny all (lowest priority)
  rule {
    action   = "deny(403)"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default deny rule"
  }

  # Allow health check IPs from Google
  rule {
    action   = "allow"
    priority = 100
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = [
          "35.191.0.0/16",
          "130.211.0.0/22",
          "209.85.152.0/22",
          "209.85.204.0/22",
        ]
      }
    }
    description = "Allow Google health checks"
  }

  # Allow specific source IPs (if configured)
  dynamic "rule" {
    for_each = length(var.allowed_source_ips) > 0 ? [1] : []
    content {
      action   = "allow"
      priority = 200
      match {
        versioned_expr = "SRC_IPS_V1"
        config {
          src_ip_ranges = var.allowed_source_ips
        }
      }
      description = "Allow configured source IPs"
    }
  }

  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = 1000
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = var.rate_limit_threshold
        interval_sec = var.rate_limit_interval_sec
      }
      ban_duration_sec = 300
    }
    description = "Rate limiting - ${var.rate_limit_threshold} requests per ${var.rate_limit_interval_sec}s"
  }

  # Block known bad user agents (scanners, bots)
  rule {
    action   = "deny(403)"
    priority = 1100
    match {
      expr {
        expression = <<-EOT
          request.headers['user-agent'].contains('sqlmap') ||
          request.headers['user-agent'].contains('nikto') ||
          request.headers['user-agent'].contains('nmap') ||
          request.headers['user-agent'].contains('masscan') ||
          request.headers['user-agent'].contains('zgrab')
        EOT
      }
    }
    description = "Block known malicious user agents"
  }

  # Block requests with suspicious patterns
  rule {
    action   = "deny(403)"
    priority = 1200
    match {
      expr {
        expression = <<-EOT
          request.path.contains('../') ||
          request.path.contains('..\\') ||
          request.path.contains('/etc/passwd') ||
          request.path.contains('/proc/self')
        EOT
      }
    }
    description = "Block path traversal attempts"
  }
}

# OWASP ModSecurity Core Rule Set (preconfigured WAF rules)
resource "google_compute_security_policy" "owasp_policy" {
  count    = var.enable_cloud_armor && var.enable_owasp_rules ? 1 : 0
  provider = google-beta

  name        = "${local.policy_name}-owasp"
  project     = var.project_id
  description = "OWASP rules for Sentinel API (${var.environment})"

  # SQL Injection protection
  rule {
    action   = "deny(403)"
    priority = 2000
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('sqli-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "SQL Injection protection (OWASP)"
  }

  # Cross-site Scripting (XSS) protection
  rule {
    action   = "deny(403)"
    priority = 2100
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('xss-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "XSS protection (OWASP)"
  }

  # Local File Inclusion protection
  rule {
    action   = "deny(403)"
    priority = 2200
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('lfi-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "Local File Inclusion protection (OWASP)"
  }

  # Remote File Inclusion protection
  rule {
    action   = "deny(403)"
    priority = 2300
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('rfi-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "Remote File Inclusion protection (OWASP)"
  }

  # Remote Code Execution protection
  rule {
    action   = "deny(403)"
    priority = 2400
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('rce-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "Remote Code Execution protection (OWASP)"
  }

  # Method enforcement (only allow expected methods)
  rule {
    action   = "deny(403)"
    priority = 2500
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('methodenforcement-v33-stable')"
      }
    }
    description = "HTTP Method enforcement (OWASP)"
  }

  # Scanner detection
  rule {
    action   = "deny(403)"
    priority = 2600
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('scannerdetection-v33-stable')"
      }
    }
    description = "Scanner detection (OWASP)"
  }

  # Protocol attack protection
  rule {
    action   = "deny(403)"
    priority = 2700
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('protocolattack-v33-stable', {'sensitivity': ${var.owasp_sensitivity_level}})"
      }
    }
    description = "Protocol attack protection (OWASP)"
  }

  # Default allow rule for OWASP policy
  rule {
    action   = "allow"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow after OWASP checks"
  }
}

################################################################################
# Secret Manager Secrets
################################################################################

resource "google_secret_manager_secret" "secrets" {
  for_each = var.create_secrets ? var.secrets : {}

  secret_id = "${each.key}-${var.environment}"
  project   = var.project_id

  labels = merge(local.labels, each.value.labels, {
    secret_name = each.key
  })

  replication {
    dynamic "user_managed" {
      for_each = length(var.secret_replication_locations) > 0 ? [1] : []
      content {
        dynamic "replicas" {
          for_each = var.secret_replication_locations
          content {
            location = replicas.value
          }
        }
      }
    }

    dynamic "auto" {
      for_each = length(var.secret_replication_locations) == 0 ? [1] : []
      content {}
    }
  }
}

################################################################################
# KMS Keyring and Keys
################################################################################

resource "google_kms_key_ring" "keyring" {
  count = var.enable_kms ? 1 : 0

  name     = "${var.kms_keyring_name}-${var.environment}"
  project  = var.project_id
  location = var.region
}

# Encryption key for storage buckets
resource "google_kms_crypto_key" "storage_key" {
  count = var.enable_kms ? 1 : 0

  name     = "storage-encryption-key"
  key_ring = google_kms_key_ring.keyring[0].id
  purpose  = "ENCRYPT_DECRYPT"

  rotation_period = var.kms_key_rotation_period

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "SOFTWARE"
  }

  labels = local.labels

  lifecycle {
    prevent_destroy = true
  }
}

# Encryption key for secrets
resource "google_kms_crypto_key" "secrets_key" {
  count = var.enable_kms ? 1 : 0

  name     = "secrets-encryption-key"
  key_ring = google_kms_key_ring.keyring[0].id
  purpose  = "ENCRYPT_DECRYPT"

  rotation_period = var.kms_key_rotation_period

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "SOFTWARE"
  }

  labels = local.labels

  lifecycle {
    prevent_destroy = true
  }
}

# Grant Cloud Storage service account access to KMS key
resource "google_kms_crypto_key_iam_member" "storage_service_account" {
  count = var.enable_kms ? 1 : 0

  crypto_key_id = google_kms_crypto_key.storage_key[0].id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com"
}

data "google_project" "project" {
  project_id = var.project_id
}
