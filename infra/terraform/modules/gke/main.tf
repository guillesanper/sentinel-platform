################################################################################
# GKE Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Creates a private GKE cluster with:
# - Private cluster (no public endpoint)
# - Workload Identity enabled
# - Network Policy enabled
# - Node pools with taints for ML workloads
# - Autoscaling
# - Shielded nodes for security
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
  cluster_name = "${var.cluster_name}-${var.environment}"

  default_labels = {
    project     = "sentinel"
    environment = var.environment
    managed_by  = "terraform"
  }
  labels = merge(local.default_labels, var.labels)

  # Workload Identity pool
  workload_identity_pool = "${var.project_id}.svc.id.goog"
}

################################################################################
# GKE Cluster
################################################################################

resource "google_container_cluster" "cluster" {
  provider = google-beta

  name     = local.cluster_name
  project  = var.project_id
  location = var.region

  # Use release channel for automatic upgrades
  release_channel {
    channel = var.release_channel
  }

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Network configuration
  network    = var.vpc_id
  subnetwork = var.subnet_id

  # IP allocation policy for VPC-native cluster
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_secondary_range_name
    services_secondary_range_name = var.services_secondary_range_name
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = var.master_cidr

    master_global_access_config {
      enabled = true
    }
  }

  # Master authorized networks
  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_cidr_blocks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_cidr_blocks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  # Workload Identity
  dynamic "workload_identity_config" {
    for_each = var.enable_workload_identity ? [1] : []
    content {
      workload_pool = local.workload_identity_pool
    }
  }

  # Network policy (Calico)
  network_policy {
    enabled  = var.enable_network_policy
    provider = var.enable_network_policy ? "CALICO" : "PROVIDER_UNSPECIFIED"
  }

  # Datapath provider for better network performance
  datapath_provider = "ADVANCED_DATAPATH"

  # Addons configuration
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = !var.enable_horizontal_pod_autoscaling
    }

    network_policy_config {
      disabled = !var.enable_network_policy
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }

    gcs_fuse_csi_driver_config {
      enabled = true
    }

    dns_cache_config {
      enabled = true
    }
  }

  # Binary Authorization
  dynamic "binary_authorization" {
    for_each = var.enable_binary_authorization ? [1] : []
    content {
      evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
    }
  }

  # Vertical Pod Autoscaling
  vertical_pod_autoscaling {
    enabled = var.enable_vertical_pod_autoscaling
  }

  # Intranode visibility for debugging
  enable_intranode_visibility = var.enable_intranode_visibility

  # Logging and monitoring
  logging_config {
    enable_components = var.logging_enabled_components
  }

  monitoring_config {
    enable_components = var.monitoring_enabled_components

    dynamic "managed_prometheus" {
      for_each = var.enable_managed_prometheus ? [1] : []
      content {
        enabled = true
      }
    }
  }

  # Maintenance window
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T${var.maintenance_start_time}:00Z"
      end_time   = "2024-01-01T${var.maintenance_end_time}:00Z"
      recurrence = var.maintenance_recurrence
    }
  }

  # Security configuration
  enable_shielded_nodes = var.enable_shielded_nodes

  # Cluster-level security posture
  security_posture_config {
    mode               = "BASIC"
    vulnerability_mode = "VULNERABILITY_BASIC"
  }

  # Resource labels
  resource_labels = local.labels

  # Deletion protection (disable for dev, enable for prod)
  deletion_protection = var.environment == "prod" ? true : false

  lifecycle {
    ignore_changes = [
      # Ignore changes to node_config since we manage node pools separately
      node_config,
    ]
  }
}

################################################################################
# Default Node Pool
################################################################################

resource "google_container_node_pool" "default" {
  name     = "default-pool"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.cluster.name

  # Node count configuration
  initial_node_count = var.default_node_pool_initial_count

  autoscaling {
    min_node_count  = var.default_node_pool_min_count
    max_node_count  = var.default_node_pool_max_count
    location_policy = "BALANCED"
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
    strategy        = "SURGE"
  }

  node_config {
    machine_type = var.default_node_pool_machine_type
    disk_size_gb = var.default_node_pool_disk_size_gb
    disk_type    = var.default_node_pool_disk_type

    # Preemptible/Spot configuration
    preemptible = var.default_node_pool_preemptible
    spot        = var.default_node_pool_spot

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    # Service account (use default compute SA if not specified)
    # In production, use a dedicated service account

    # Network tags
    tags = var.network_tags

    # Labels
    labels = merge(local.labels, {
      node_pool = "default"
    })

    # Workload metadata config for Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Confidential nodes
    dynamic "confidential_nodes" {
      for_each = var.enable_confidential_nodes ? [1] : []
      content {
        enabled = true
      }
    }

    # Kubelet config for better performance
    kubelet_config {
      cpu_manager_policy = "static"
      pod_pids_limit     = 4096
    }

    # Linux node config
    linux_node_config {
      sysctl = {
        "net.core.somaxconn"           = "32768"
        "net.ipv4.tcp_max_syn_backlog" = "32768"
      }
    }

    # GCE metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  lifecycle {
    ignore_changes = [
      initial_node_count,
    ]
  }
}

################################################################################
# ML Node Pool (with optional GPU support)
################################################################################

resource "google_container_node_pool" "ml" {
  count = var.enable_ml_node_pool ? 1 : 0

  name     = "ml-pool"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.cluster.name

  # Node count configuration
  initial_node_count = var.ml_node_pool_initial_count

  autoscaling {
    min_node_count  = var.ml_node_pool_min_count
    max_node_count  = var.ml_node_pool_max_count
    location_policy = "BALANCED"
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
    strategy        = "SURGE"
  }

  node_config {
    machine_type = var.ml_node_pool_machine_type
    disk_size_gb = var.ml_node_pool_disk_size_gb
    disk_type    = "pd-ssd"

    # Preemptible/Spot configuration
    preemptible = var.ml_node_pool_preemptible
    spot        = var.ml_node_pool_spot

    # GPU configuration
    dynamic "guest_accelerator" {
      for_each = var.ml_node_pool_gpu_type != "" ? [1] : []
      content {
        type  = var.ml_node_pool_gpu_type
        count = var.ml_node_pool_gpu_count
        gpu_driver_installation_config {
          gpu_driver_version = "LATEST"
        }
      }
    }

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    # Network tags
    tags = var.network_tags

    # Labels
    labels = merge(local.labels, {
      node_pool = "ml"
      workload  = "ml"
    })

    # Taints for ML workloads
    taint {
      key    = "workload"
      value  = "ml"
      effect = "NO_SCHEDULE"
    }

    # Workload metadata config for Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = var.ml_node_pool_gpu_type == "" ? true : false
      enable_integrity_monitoring = true
    }

    # GCE metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  lifecycle {
    ignore_changes = [
      initial_node_count,
    ]
  }
}
