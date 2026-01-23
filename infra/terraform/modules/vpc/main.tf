################################################################################
# VPC Module - Main Configuration
# Sentinel Platform - Cloud-native intrusion detection system
#
# Creates a secure VPC with:
# - Private subnets for GKE nodes, pods, and services
# - Cloud NAT for egress traffic
# - Cloud Router for NAT gateway
# - Restrictive firewall rules (deny all ingress by default)
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
  vpc_name    = "${var.vpc_name}-${var.environment}"
  subnet_name = "sentinel-subnet-${var.environment}"
  router_name = "sentinel-router-${var.environment}"
  nat_name    = "sentinel-nat-${var.environment}"

  default_labels = {
    project     = "sentinel"
    environment = var.environment
    managed_by  = "terraform"
  }
  labels = merge(local.default_labels, var.labels)
}

################################################################################
# VPC Network
################################################################################

resource "google_compute_network" "vpc" {
  name                            = local.vpc_name
  project                         = var.project_id
  auto_create_subnetworks         = false
  routing_mode                    = "REGIONAL"
  delete_default_routes_on_create = false
  mtu                             = 1460

  description = "Sentinel Platform VPC for ${var.environment} environment"
}

################################################################################
# Private Subnet for GKE
################################################################################

resource "google_compute_subnetwork" "gke_subnet" {
  name                     = local.subnet_name
  project                  = var.project_id
  region                   = var.region
  network                  = google_compute_network.vpc.id
  ip_cidr_range            = var.nodes_cidr
  private_ip_google_access = true

  description = "Private subnet for GKE nodes in ${var.environment}"

  # Secondary ranges for GKE pods and services
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  # VPC Flow Logs for security monitoring
  dynamic "log_config" {
    for_each = var.enable_flow_logs ? [1] : []
    content {
      aggregation_interval = var.flow_logs_interval
      flow_sampling        = var.flow_logs_sampling
      metadata             = "INCLUDE_ALL_METADATA"
      filter_expr          = "true"
    }
  }
}

################################################################################
# Cloud Router for NAT Gateway
################################################################################

resource "google_compute_router" "router" {
  name    = local.router_name
  project = var.project_id
  region  = var.region
  network = google_compute_network.vpc.id

  description = "Cloud Router for Sentinel NAT gateway in ${var.environment}"

  bgp {
    asn = 64514
  }
}

################################################################################
# Cloud NAT for Egress Traffic
################################################################################

resource "google_compute_router_nat" "nat" {
  name                               = local.nat_name
  project                            = var.project_id
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  # Logging for security audit
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }

  # TCP timeouts
  tcp_established_idle_timeout_sec = 1200
  tcp_transitory_idle_timeout_sec  = 30
  tcp_time_wait_timeout_sec        = 120

  # UDP timeout
  udp_idle_timeout_sec = 30

  # ICMP timeout
  icmp_idle_timeout_sec = 30

  # Limit ports per VM for security
  min_ports_per_vm                    = 64
  max_ports_per_vm                    = 2048
  enable_endpoint_independent_mapping = false
}

################################################################################
# Firewall Rules
################################################################################

# Deny all ingress by default (highest priority deny rule)
resource "google_compute_firewall" "deny_all_ingress" {
  name        = "sentinel-deny-all-ingress-${var.environment}"
  project     = var.project_id
  network     = google_compute_network.vpc.id
  priority    = 65534
  direction   = "INGRESS"
  description = "Deny all ingress traffic by default"

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# Allow internal communication within VPC
resource "google_compute_firewall" "allow_internal" {
  name        = "sentinel-allow-internal-${var.environment}"
  project     = var.project_id
  network     = google_compute_network.vpc.id
  priority    = 1000
  direction   = "INGRESS"
  description = "Allow internal communication within VPC"

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [
    var.nodes_cidr,
    var.pods_cidr,
    var.services_cidr,
  ]

  target_tags = ["sentinel-node"]
}

# Allow GKE master to communicate with nodes (webhook, kubelet)
resource "google_compute_firewall" "allow_master_to_nodes" {
  name        = "sentinel-allow-master-${var.environment}"
  project     = var.project_id
  network     = google_compute_network.vpc.id
  priority    = 1000
  direction   = "INGRESS"
  description = "Allow GKE master to communicate with nodes"

  allow {
    protocol = "tcp"
    ports    = ["443", "10250", "8443"]
  }

  source_ranges = [var.master_cidr]
  target_tags   = ["sentinel-node"]
}

# Allow health checks from Google Cloud Load Balancers
resource "google_compute_firewall" "allow_health_checks" {
  name        = "sentinel-allow-health-checks-${var.environment}"
  project     = var.project_id
  network     = google_compute_network.vpc.id
  priority    = 1000
  direction   = "INGRESS"
  description = "Allow health checks from GCP load balancers"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080", "8000"]
  }

  # Google Cloud health check ranges
  source_ranges = [
    "35.191.0.0/16",
    "130.211.0.0/22",
    "209.85.152.0/22",
    "209.85.204.0/22",
  ]

  target_tags = ["sentinel-node"]
}

# Allow IAP for secure SSH access (if needed for debugging)
resource "google_compute_firewall" "allow_iap_ssh" {
  name        = "sentinel-allow-iap-ssh-${var.environment}"
  project     = var.project_id
  network     = google_compute_network.vpc.id
  priority    = 1000
  direction   = "INGRESS"
  description = "Allow SSH via IAP tunnel for secure debugging"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # IAP's IP range
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["sentinel-node"]
}

################################################################################
# Private Service Connection (for Cloud SQL, etc.)
################################################################################

resource "google_compute_global_address" "private_ip_range" {
  name          = "sentinel-private-ip-range-${var.environment}"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
  description   = "Private IP range for Google managed services"
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]

  deletion_policy = "ABANDON"
}
