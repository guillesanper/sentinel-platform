################################################################################
# GKE Module Variables
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the GKE cluster"
  type        = string
}

variable "zones" {
  description = "GCP zones for the GKE node pools"
  type        = list(string)
  default     = []
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "sentinel-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster"
  type        = string
  default     = "1.28"
}

variable "release_channel" {
  description = "Release channel for GKE version upgrades"
  type        = string
  default     = "REGULAR"
  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE", "UNSPECIFIED"], var.release_channel)
    error_message = "Release channel must be one of: RAPID, REGULAR, STABLE, UNSPECIFIED."
  }
}

################################################################################
# Network Configuration
################################################################################

variable "vpc_id" {
  description = "VPC network ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for GKE nodes"
  type        = string
}

variable "pods_secondary_range_name" {
  description = "Name of the secondary range for pods"
  type        = string
}

variable "services_secondary_range_name" {
  description = "Name of the secondary range for services"
  type        = string
}

variable "master_cidr" {
  description = "CIDR range for GKE master nodes"
  type        = string
  default     = "172.16.0.0/28"
}

variable "master_authorized_cidr_blocks" {
  description = "CIDR blocks authorized to access the master endpoint"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "enable_private_endpoint" {
  description = "Enable private endpoint (no public IP for master)"
  type        = bool
  default     = true
}

variable "enable_private_nodes" {
  description = "Enable private nodes (no public IPs)"
  type        = bool
  default     = true
}

################################################################################
# Node Pool Configuration - Default Pool
################################################################################

variable "default_node_pool_machine_type" {
  description = "Machine type for default node pool"
  type        = string
  default     = "n2-standard-4"
}

variable "default_node_pool_min_count" {
  description = "Minimum number of nodes in default pool"
  type        = number
  default     = 1
}

variable "default_node_pool_max_count" {
  description = "Maximum number of nodes in default pool"
  type        = number
  default     = 10
}

variable "default_node_pool_initial_count" {
  description = "Initial number of nodes in default pool"
  type        = number
  default     = 2
}

variable "default_node_pool_disk_size_gb" {
  description = "Disk size in GB for default pool nodes"
  type        = number
  default     = 100
}

variable "default_node_pool_disk_type" {
  description = "Disk type for default pool nodes"
  type        = string
  default     = "pd-ssd"
}

variable "default_node_pool_preemptible" {
  description = "Use preemptible VMs for default pool"
  type        = bool
  default     = false
}

variable "default_node_pool_spot" {
  description = "Use spot VMs for default pool"
  type        = bool
  default     = false
}

################################################################################
# Node Pool Configuration - ML Pool
################################################################################

variable "enable_ml_node_pool" {
  description = "Enable dedicated ML node pool"
  type        = bool
  default     = true
}

variable "ml_node_pool_machine_type" {
  description = "Machine type for ML node pool"
  type        = string
  default     = "n2-highmem-4"
}

variable "ml_node_pool_min_count" {
  description = "Minimum number of nodes in ML pool"
  type        = number
  default     = 0
}

variable "ml_node_pool_max_count" {
  description = "Maximum number of nodes in ML pool"
  type        = number
  default     = 5
}

variable "ml_node_pool_initial_count" {
  description = "Initial number of nodes in ML pool"
  type        = number
  default     = 1
}

variable "ml_node_pool_disk_size_gb" {
  description = "Disk size in GB for ML pool nodes"
  type        = number
  default     = 200
}

variable "ml_node_pool_preemptible" {
  description = "Use preemptible VMs for ML pool"
  type        = bool
  default     = false
}

variable "ml_node_pool_spot" {
  description = "Use spot VMs for ML pool"
  type        = bool
  default     = false
}

variable "ml_node_pool_gpu_type" {
  description = "GPU type for ML nodes (empty for no GPU)"
  type        = string
  default     = ""
}

variable "ml_node_pool_gpu_count" {
  description = "Number of GPUs per ML node"
  type        = number
  default     = 0
}

################################################################################
# Security Configuration
################################################################################

variable "enable_workload_identity" {
  description = "Enable Workload Identity for GKE"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Kubernetes Network Policy"
  type        = bool
  default     = true
}

variable "enable_shielded_nodes" {
  description = "Enable Shielded GKE Nodes"
  type        = bool
  default     = true
}

variable "enable_binary_authorization" {
  description = "Enable Binary Authorization"
  type        = bool
  default     = false
}

variable "enable_confidential_nodes" {
  description = "Enable Confidential GKE Nodes (AMD SEV)"
  type        = bool
  default     = false
}

variable "enable_intranode_visibility" {
  description = "Enable intranode visibility for network debugging"
  type        = bool
  default     = true
}

################################################################################
# Maintenance and Operations
################################################################################

variable "maintenance_start_time" {
  description = "Start time for maintenance window (UTC)"
  type        = string
  default     = "03:00"
}

variable "maintenance_end_time" {
  description = "End time for maintenance window (UTC)"
  type        = string
  default     = "07:00"
}

variable "maintenance_recurrence" {
  description = "Recurrence rule for maintenance window"
  type        = string
  default     = "FREQ=WEEKLY;BYDAY=SA,SU"
}

variable "enable_vertical_pod_autoscaling" {
  description = "Enable Vertical Pod Autoscaling"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaling" {
  description = "Enable cluster-level Horizontal Pod Autoscaling"
  type        = bool
  default     = true
}

################################################################################
# Logging and Monitoring
################################################################################

variable "logging_enabled_components" {
  description = "GKE components to enable logging for"
  type        = list(string)
  default     = ["SYSTEM_COMPONENTS", "WORKLOADS"]
}

variable "monitoring_enabled_components" {
  description = "GKE components to enable monitoring for"
  type        = list(string)
  default     = ["SYSTEM_COMPONENTS"]
}

variable "enable_managed_prometheus" {
  description = "Enable Google Cloud Managed Prometheus"
  type        = bool
  default     = false
}

################################################################################
# Labels
################################################################################

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "network_tags" {
  description = "Network tags to apply to GKE nodes"
  type        = list(string)
  default     = ["sentinel-node"]
}
