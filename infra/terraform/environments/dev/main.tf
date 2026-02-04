################################################################################
# Dev Environment - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Development environment with:
# - Smaller node pools (cost optimization)
# - Single zone deployment
# - Preemptible/spot instances
# - Relaxed security for development
################################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubectl = {
      source  = "alekc/kubectl"
      version = "~> 2.0"
    }
  }

  # Backend configuration - uncomment and configure for remote state
  # backend "gcs" {
  #   bucket = "sentinel-terraform-state"
  #   prefix = "dev"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Get GKE cluster credentials
data "google_client_config" "default" {}

# Kubernetes provider - configured after GKE cluster is created
provider "kubernetes" {
  host                   = "https://${module.gke.cluster_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${module.gke.cluster_endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(module.gke.cluster_ca_certificate)
  }
}

provider "kubectl" {
  host                   = "https://${module.gke.cluster_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.cluster_ca_certificate)
  load_config_file       = false
}

locals {
  environment = "dev"

  labels = {
    project     = "sentinel"
    environment = local.environment
    managed_by  = "terraform"
    cost_center = "development"
  }
}

################################################################################
# VPC Module
################################################################################

module "vpc" {
  source = "../../modules/vpc"

  project_id  = var.project_id
  region      = var.region
  environment = local.environment

  vpc_name = "sentinel-vpc"

  # Network CIDRs
  nodes_cidr    = "10.0.0.0/20"
  pods_cidr     = "10.4.0.0/14"
  services_cidr = "10.8.0.0/20"
  master_cidr   = "172.16.0.0/28"

  # Enable flow logs for debugging
  enable_flow_logs   = true
  flow_logs_sampling = 0.5

  labels = local.labels
}

################################################################################
# GKE Module
################################################################################

module "gke" {
  source = "../../modules/gke"

  project_id  = var.project_id
  region      = var.region
  environment = local.environment

  cluster_name       = "sentinel-cluster"
  kubernetes_version = "1.28"
  release_channel    = "REGULAR"

  # Network configuration
  vpc_id                        = module.vpc.vpc_id
  subnet_id                     = module.vpc.subnet_id
  pods_secondary_range_name     = module.vpc.pods_secondary_range_name
  services_secondary_range_name = module.vpc.services_secondary_range_name
  master_cidr                   = module.vpc.master_cidr

  # Private cluster but with accessible endpoint for dev
  enable_private_endpoint = false  # Allow external access in dev
  enable_private_nodes    = true

  # Master authorized networks (add your IP for kubectl access)
  master_authorized_cidr_blocks = [
    {
      cidr_block   = "0.0.0.0/0"  # WARNING: Open for dev only
      display_name = "All networks (dev only)"
    }
  ]

  # Default node pool - smaller for dev
  default_node_pool_machine_type  = "n2-standard-2"
  default_node_pool_min_count     = 1
  default_node_pool_max_count     = 3
  default_node_pool_initial_count = 1
  default_node_pool_disk_size_gb  = 50
  default_node_pool_disk_type     = "pd-balanced"  # Avoid SSD quota limits
  default_node_pool_spot          = true  # Use spot for cost savings

  # ML node pool - disabled for dev (saves SSD quota)
  enable_ml_node_pool = false

  # Security features
  enable_workload_identity   = true
  enable_network_policy      = true
  enable_shielded_nodes      = true
  enable_confidential_nodes  = false  # Not needed for dev
  enable_binary_authorization = false  # Disabled for dev

  # Autoscaling
  enable_vertical_pod_autoscaling   = true
  enable_horizontal_pod_autoscaling = true

  # Monitoring
  logging_enabled_components    = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  monitoring_enabled_components = ["SYSTEM_COMPONENTS"]
  enable_managed_prometheus     = false  # Use self-managed in dev

  network_tags = module.vpc.network_tags
  labels       = local.labels

  depends_on = [module.vpc]
}

################################################################################
# Storage Module
################################################################################

module "storage" {
  source = "../../modules/storage"

  project_id  = var.project_id
  region      = var.region
  environment = local.environment

  # Models bucket
  models_bucket_name        = "sentinel-models"
  models_versioning_enabled = true
  models_retention_days     = 30  # Shorter retention for dev

  # Logs bucket
  logs_bucket_name      = "sentinel-logs"
  logs_retention_days   = 7   # Short retention for dev
  logs_archive_days     = 14
  logs_delete_days      = 30

  # Data bucket
  enable_data_bucket        = true
  data_bucket_name          = "sentinel-data"
  data_versioning_enabled   = false  # No versioning in dev

  # Security - relaxed for dev
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  enable_cmek                 = false  # No CMEK in dev

  labels = local.labels
}

################################################################################
# IAM Module
################################################################################

module "iam" {
  source = "../../modules/iam"

  project_id  = var.project_id
  environment = local.environment

  # Workload Identity
  enable_workload_identity = true
  gke_cluster_name         = module.gke.cluster_name

  # K8s service accounts
  api_k8s_namespace       = "platform"
  api_k8s_service_account = "sentinel-api"
  ml_k8s_namespace        = "ml"
  ml_k8s_service_account  = "sentinel-ml"

  # Storage buckets
  models_bucket_name = module.storage.models_bucket_name
  logs_bucket_name   = module.storage.logs_bucket_name
  data_bucket_name   = module.storage.data_bucket_name

  # Custom roles
  create_custom_roles = true

  labels = local.labels

  depends_on = [module.gke, module.storage]
}

################################################################################
# Security Module
################################################################################

module "security" {
  source = "../../modules/security"

  project_id  = var.project_id
  region      = var.region
  environment = local.environment

  # Cloud Armor - disabled for dev (cost savings)
  enable_cloud_armor = false

  # Secret Manager
  create_secrets = true
  secrets = {
    jwt-secret = {
      description = "JWT signing secret"
    }
    api-key = {
      description = "API key for service auth"
    }
  }

  # KMS - disabled for dev
  enable_kms = false

  labels = local.labels
}

################################################################################
# Kubernetes Module - External Secrets Operator
################################################################################

module "kubernetes" {
  source = "../../modules/kubernetes"

  project_id  = var.project_id
  region      = var.region
  environment = local.environment

  # GKE cluster configuration
  cluster_name           = module.gke.cluster_name
  cluster_endpoint       = module.gke.cluster_endpoint
  cluster_ca_certificate = module.gke.cluster_ca_certificate

  # External Secrets Operator
  install_eso   = true
  eso_namespace = "external-secrets"

  # Secrets to sync from GCP Secret Manager
  secrets_to_sync   = ["jwt-secret", "api-key"]
  target_namespaces = ["platform", "ml"]

  labels = local.labels

  depends_on = [module.gke, module.security, module.iam]
}
