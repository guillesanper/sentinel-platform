################################################################################
# VPC Module Outputs
# Sentinel Platform - Cloud-native intrusion detection system
################################################################################

output "vpc_id" {
  description = "The ID of the VPC"
  value       = google_compute_network.vpc.id
}

output "vpc_name" {
  description = "The name of the VPC"
  value       = google_compute_network.vpc.name
}

output "vpc_self_link" {
  description = "The self link of the VPC"
  value       = google_compute_network.vpc.self_link
}

output "subnet_id" {
  description = "The ID of the GKE subnet"
  value       = google_compute_subnetwork.gke_subnet.id
}

output "subnet_name" {
  description = "The name of the GKE subnet"
  value       = google_compute_subnetwork.gke_subnet.name
}

output "subnet_self_link" {
  description = "The self link of the GKE subnet"
  value       = google_compute_subnetwork.gke_subnet.self_link
}

output "subnet_cidr" {
  description = "The CIDR range of the GKE subnet"
  value       = google_compute_subnetwork.gke_subnet.ip_cidr_range
}

output "pods_secondary_range_name" {
  description = "The name of the secondary range for pods"
  value       = "pods"
}

output "services_secondary_range_name" {
  description = "The name of the secondary range for services"
  value       = "services"
}

output "pods_cidr" {
  description = "The CIDR range for pods"
  value       = var.pods_cidr
}

output "services_cidr" {
  description = "The CIDR range for services"
  value       = var.services_cidr
}

output "master_cidr" {
  description = "The CIDR range for GKE master nodes"
  value       = var.master_cidr
}

output "router_name" {
  description = "The name of the Cloud Router"
  value       = google_compute_router.router.name
}

output "nat_name" {
  description = "The name of the Cloud NAT"
  value       = google_compute_router_nat.nat.name
}

output "private_vpc_connection" {
  description = "The private VPC connection for Google managed services"
  value       = google_service_networking_connection.private_vpc_connection.id
}

output "network_tags" {
  description = "Network tags for GKE nodes"
  value       = ["sentinel-node"]
}
