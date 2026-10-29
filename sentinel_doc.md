# Sentinel: Cloud-Native Intrusion Detection Platform

## Technical Design Document v1.0

---

# Executive Summary

**Sentinel** es una plataforma cloud-native de detección de intrusiones basada en Machine Learning, diseñada para identificar comportamientos maliciosos y anomalías en logs de infraestructura en tiempo real.

**Visión:** Diseñar, desplegar y operar una plataforma de seguridad cloud-native en GCP que demuestre dominio de infraestructura moderna, MLOps, DevSecOps y observabilidad.

**Stack Principal:**
- **Cloud:** Google Cloud Platform
- **IaC:** Terraform
- **Orquestación:** Kubernetes (GKE)
- **API:** FastAPI
- **ML:** scikit-learn, XGBoost, MLflow
- **Observabilidad:** Prometheus, Grafana
- **CI/CD:** GitHub Actions

---

# Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Architecture](#2-solution-architecture)
3. [Infrastructure Layer (Terraform)](#3-infrastructure-layer)
4. [Data Pipeline](#4-data-pipeline)
5. [ML/AI System](#5-mlai-system)
6. [API Layer](#6-api-layer)
7. [Kubernetes Platform](#7-kubernetes-platform)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Observability Stack](#9-observability-stack)
10. [Security Architecture](#10-security-architecture)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Repository Structure](#12-repository-structure)
13. [Getting Started](#13-getting-started)

---

# 1. Problem Statement

## 1.1 El Desafío

Las organizaciones generan millones de eventos de log diariamente. Detectar actividad maliciosa manualmente es imposible y los sistemas basados en reglas estáticas fallan ante ataques novedosos.

## 1.2 Tipos de Amenazas a Detectar

| Categoría | Ejemplos | Indicadores |
|-----------|----------|-------------|
| **Brute Force** | SSH, login attempts | >5 failed logins/min mismo IP |
| **Credential Stuffing** | Multiple users/IPs | >10 failed logins/min diferentes IPs |
| **Privilege Escalation** | sudo abuse, role changes | Usuario normal → root inesperado |
| **Lateral Movement** | Internal pivoting | Conexiones internas inusuales |
| **Data Exfiltration** | Large transfers | >1GB egress en horas inusuales |
| **Reconnaissance** | Port scanning | >100 puertos/min desde un IP |

## 1.3 Requisitos

**Funcionales:**
- Ingestar logs de múltiples fuentes (auth, system, network, application)
- Detectar patrones de ataque conocidos en tiempo real
- Identificar anomalías comportamentales mediante ML
- Exponer predicciones vía API REST
- Generar alertas priorizadas por severidad

**No Funcionales:**
- Latencia de predicción < 100ms (p99)
- Disponibilidad 99.9%
- Escalado automático
- Seguro por diseño (zero trust)

---

# 2. Solution Architecture

## 2.1 High-Level Architecture

```
                                    INTERNET
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   Cloud Armor   │  ← WAF + DDoS
                              │   (L7 Firewall) │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  Cloud Load     │  ← HTTPS LB
                              │  Balancer       │
                              └────────┬────────┘
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                           GKE PRIVATE CLUSTER                                 │
│                                                                               │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │    PLATFORM     │    │       ML        │    │  OBSERVABILITY  │         │
│   │   NAMESPACE     │    │   NAMESPACE     │    │   NAMESPACE     │         │
│   │                 │    │                 │    │                 │         │
│   │ • Ingestion     │    │ • Prediction    │    │ • Prometheus    │         │
│   │ • Feature Eng.  │◄──►│   API (FastAPI) │    │ • Grafana       │         │
│   │                 │    │ • Model Server  │    │ • Alertmanager  │         │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                               │
        ▼                              ▼                               ▼
┌───────────────┐            ┌─────────────────┐            ┌─────────────────┐
│ Cloud Storage │            │ Secret Manager  │            │ Cloud Monitoring│
│ • Raw logs    │            │ • API keys      │            │ • Metrics       │
│ • Models      │            │ • Credentials   │            │ • Logs          │
│ • Features    │            │                 │            │                 │
└───────────────┘            └─────────────────┘            └─────────────────┘
```

## 2.2 Data Flow

```
LOG SOURCES              INGESTION           PROCESSING            OUTPUT
───────────              ─────────           ──────────            ──────

┌─────────────┐
│ Auth Logs   │──┐
│ (SSH, sudo) │  │
└─────────────┘  │       ┌──────────┐      ┌──────────┐      ┌──────────┐
                 │       │          │      │          │      │          │
┌─────────────┐  ├──────►│  Pub/Sub │─────►│ Feature  │─────►│ ML Model │
│ System Logs │──┤       │          │      │ Pipeline │      │ Predict  │
│ (syslog)    │  │       └──────────┘      └──────────┘      └────┬─────┘
└─────────────┘  │                                                 │
                 │                                                 ▼
┌─────────────┐  │                                          ┌──────────┐
│ Network     │──┤                                          │  Alert   │
│ (firewall)  │  │                                          │  System  │
└─────────────┘  │                                          └──────────┘
                 │
┌─────────────┐  │
│ App Logs    │──┘
└─────────────┘
```

## 2.3 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Cloud** | GCP | Managed services, GKE maturity |
| **IaC** | Terraform | Industry standard |
| **Container** | GKE (Kubernetes) | Production-grade orchestration |
| **API** | FastAPI + Uvicorn | Async, OpenAPI native |
| **ML Training** | scikit-learn, XGBoost | Interpretable models |
| **ML Ops** | MLflow | Experiment tracking, registry |
| **Data** | Pandas, GCS | Simple, cost-effective |
| **Observability** | Prometheus + Grafana | Cloud-native standard |
| **CI/CD** | GitHub Actions | Native integration |
| **Security** | Trivy, Semgrep | Comprehensive scanning |

---

# 3. Infrastructure Layer

## 3.1 GCP Resources Overview

```
GCP Project: sentinel-platform
│
├── Networking
│   ├── VPC: sentinel-vpc (10.0.0.0/16)
│   ├── Subnet: sentinel-gke-nodes (10.0.1.0/24) - Private
│   ├── Subnet: sentinel-gke-pods (10.0.16.0/20) - Secondary
│   ├── Subnet: sentinel-gke-services (10.0.32.0/20) - Secondary
│   ├── Cloud NAT: sentinel-nat
│   └── Cloud Router: sentinel-router
│
├── Compute
│   └── GKE Cluster: sentinel-cluster (Private)
│       ├── Node Pool: default (e2-standard-4)
│       └── Node Pool: ml-pool (n1-standard-8, GPU optional)
│
├── Storage
│   ├── GCS: sentinel-data-raw
│   ├── GCS: sentinel-data-processed
│   ├── GCS: sentinel-models
│   └── GCS: sentinel-mlflow
│
├── Security
│   ├── Secret Manager: API keys, credentials
│   ├── Cloud Armor: WAF policies
│   └── IAM: Service accounts, roles
│
└── Monitoring
    ├── Cloud Logging
    └── Cloud Monitoring
```

## 3.2 Terraform Module Structure

```
infra/terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
│
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── gke/
│   ├── storage/
│   ├── iam/
│   └── security/
│
└── backend.tf
```

## 3.3 VPC Configuration

```hcl
# modules/vpc/main.tf

resource "google_compute_network" "sentinel" {
  name                    = "sentinel-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "gke_nodes" {
  name          = "sentinel-gke-nodes"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.sentinel.id

  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.0.16.0/20"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.0.32.0/20"
  }
}

# Cloud NAT for outbound internet access
resource "google_compute_router" "sentinel" {
  name    = "sentinel-router"
  region  = var.region
  network = google_compute_network.sentinel.id
}

resource "google_compute_router_nat" "sentinel" {
  name                               = "sentinel-nat"
  router                             = google_compute_router.sentinel.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "sentinel-allow-internal"
  network = google_compute_network.sentinel.name

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

  source_ranges = ["10.0.0.0/16"]
}

resource "google_compute_firewall" "deny_all_ingress" {
  name     = "sentinel-deny-all-ingress"
  network  = google_compute_network.sentinel.name
  priority = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
}
```

## 3.4 GKE Cluster Configuration

```hcl
# modules/gke/main.tf

resource "google_container_cluster" "sentinel" {
  name     = "sentinel-cluster"
  location = var.region

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # Private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Networking
  network    = var.network_id
  subnetwork = var.subnetwork_id

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Security
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Logging & Monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }
}

# Node pool
resource "google_container_node_pool" "primary" {
  name       = "primary-pool"
  location   = var.region
  cluster    = google_container_cluster.sentinel.name
  node_count = var.node_count

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    # Security
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      team        = "platform"
    }
  }

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
```

## 3.5 Storage Configuration

```hcl
# modules/storage/main.tf

locals {
  buckets = {
    raw = {
      name     = "${var.project_id}-data-raw"
      location = var.region
      lifecycle_age = 90
    }
    processed = {
      name     = "${var.project_id}-data-processed"
      location = var.region
      lifecycle_age = 365
    }
    models = {
      name     = "${var.project_id}-models"
      location = var.region
      lifecycle_age = null  # No auto-delete
    }
    mlflow = {
      name     = "${var.project_id}-mlflow"
      location = var.region
      lifecycle_age = null
    }
  }
}

resource "google_storage_bucket" "buckets" {
  for_each = local.buckets

  name     = each.value.name
  location = each.value.location

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  dynamic "lifecycle_rule" {
    for_each = each.value.lifecycle_age != null ? [1] : []
    content {
      condition {
        age = each.value.lifecycle_age
      }
      action {
        type = "Delete"
      }
    }
  }

  encryption {
    default_kms_key_name = var.kms_key_id
  }
}
```

## 3.6 IAM Configuration

```hcl
# modules/iam/main.tf

# Service Accounts
resource "google_service_account" "api" {
  account_id   = "sentinel-api"
  display_name = "Sentinel Prediction API"
}

resource "google_service_account" "training" {
  account_id   = "sentinel-training"
  display_name = "Sentinel ML Training"
}

# API Service Account Permissions
resource "google_project_iam_member" "api_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.api.email}"

  condition {
    title      = "only_models_bucket"
    expression = "resource.name.startsWith(\"projects/_/buckets/${var.project_id}-models\")"
  }
}

resource "google_project_iam_member" "api_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Training Service Account Permissions
resource "google_project_iam_member" "training_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.training.email}"
}

# Workload Identity Bindings
resource "google_service_account_iam_member" "api_workload_identity" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[sentinel-ml/prediction-api-sa]"
}
```

---

# 4. Data Pipeline

Sentinel ingiere **dos modalidades de dato**, con esquemas y pipelines separados:

| Modalidad | Esquema | Pipeline de features | Track |
|---|---|---|---|
| Evento de log | [`data/schemas/event.json`](./data/schemas/event.json) | `ml/features/pipeline.py` (25 features derivadas) | §5.1–§5.6 |
| Flujo de red | [`data/schemas/flow.json`](./data/schemas/flow.json) | `ml/flow/features.py` (78 columnas CICFlowMeter) | §5.7 |

El resto de §4 describe la modalidad de **evento**. Para la de flujo, el JSON Schema (draft
2020-12) de `flow.json` define un objeto `features` con las 78 columnas —todas opcionales,
ausente → 0.0— más metadatos opcionales (`src_ip`, `dst_ip`, `dst_port`, `protocol`) que **quedan
fuera del vector del modelo** y solo alimentan la explicación y el log. Esa separación es
estructural, no una denylist: nada fuera de `features` puede llegar al modelo, así que añadir un
campo de contexto no puede contaminarlo por olvido. Un payload cuyas claves no incluyan **ningún**
nombre reconocido se rechaza con 422, para que un cliente con el esquema equivocado no reciba un
vector de ceros puntuado como flujo real.

## 4.1 Log Schema (Unified)

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2024-01-15T14:32:18.123Z",
  "source": {
    "type": "auth|system|network|app",
    "hostname": "web-server-01",
    "ip": "10.0.1.45",
    "service": "sshd"
  },
  "actor": {
    "user": "admin",
    "uid": 1000,
    "source_ip": "203.0.113.45",
    "geo": {
      "country": "CN",
      "city": "Beijing"
    }
  },
  "action": {
    "type": "authentication",
    "subtype": "login_failed",
    "result": "failure",
    "reason": "invalid_password"
  },
  "raw": "Jan 15 14:32:18 web-server-01 sshd[12345]: Failed password..."
}
```

## 4.2 Attack Patterns by Log Type

### Authentication Logs
| Pattern | Indicators | Severity |
|---------|------------|----------|
| Brute Force | >5 failed logins/min same IP | HIGH |
| Credential Stuffing | >10 failed logins/min diff IPs | HIGH |
| Password Spraying | 1-2 attempts per user, many users | MEDIUM |
| Success after failures | Success following >3 failures | HIGH |
| Root login unusual IP | root + new source IP | CRITICAL |

### System Logs
| Pattern | Indicators | Severity |
|---------|------------|----------|
| Privilege escalation | User → root unexpectedly | CRITICAL |
| Kernel module loading | insmod/modprobe unusual | HIGH |
| File integrity changes | /etc/passwd modified | CRITICAL |
| Service manipulation | systemctl stop firewall | HIGH |

### Network Logs
| Pattern | Indicators | Severity |
|---------|------------|----------|
| Port scanning | >100 ports/min single IP | MEDIUM |
| Data exfiltration | >1GB egress unusual hours | HIGH |
| C2 beaconing | Regular interval connections | CRITICAL |
| Lateral movement | Internal→Internal unusual | HIGH |

## 4.3 Storage Structure

```
gs://sentinel-data-raw/
├── auth/year=2024/month=01/day=15/
├── system/year=2024/month=01/day=15/
├── network/year=2024/month=01/day=15/
└── application/year=2024/month=01/day=15/

gs://sentinel-data-processed/
└── unified/year=2024/month=01/day=15/
    └── events_20240115.parquet

gs://sentinel-models/
├── anomaly_detector/v1/
├── anomaly_detector/v2/
└── attack_classifier/v1/

gs://sentinel-mlflow/
├── experiments/
└── artifacts/
```

## 4.4 Feature Engineering

### Features Extracted

| Feature | Type | Description | Attack Relevance |
|---------|------|-------------|------------------|
| `hour_of_day` | int | 0-23 | Off-hours attacks |
| `is_night` | bool | 00:00-06:00 | Unauthorized timing |
| `user_failed_logins_1h` | int | Failures by user | Brute force |
| `ip_events_last_1min` | int | Events from IP | Rate attacks |
| `ip_unique_users_attempted` | int | Users from IP | Credential stuffing |
| `ip_geo_distance_km` | float | Distance from usual | Impossible travel |
| `ip_is_tor_exit` | bool | Tor exit node | Anonymization |
| `action_is_privileged` | bool | root/admin action | Privilege abuse |
| `success_after_failures` | int | Success after N fails | Compromised account |
| `events_per_minute_zscore` | float | Rate anomaly | Behavioral anomaly |

### Feature Pipeline Code

```python
# ml/features/pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class FeatureExtractor:
    """Extract security-relevant features from log events."""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client  # For rolling window calculations
    
    def transform(self, event: dict) -> pd.DataFrame:
        """Transform single event to feature vector."""
        features = {
            # Temporal
            'hour_of_day': self._extract_hour(event),
            'is_weekend': self._is_weekend(event),
            'is_night': self._is_night(event),
            
            # User behavior (rolling windows)
            'user_failed_logins_1h': self._user_failures(event, hours=1),
            'user_events_last_5min': self._user_events(event, minutes=5),
            'user_unique_ips_24h': self._user_unique_ips(event, hours=24),
            
            # Source IP
            'ip_events_last_1min': self._ip_events(event, minutes=1),
            'ip_unique_users': self._ip_unique_users(event),
            'ip_is_internal': self._is_internal(event),
            'ip_geo_distance_km': self._geo_distance(event),
            'ip_is_tor_exit': self._is_tor(event),
            
            # Action
            'action_is_privileged': self._is_privileged(event),
            'action_success': event['action']['result'] == 'success',
            
            # Sequence
            'success_after_failures': self._success_after_failures(event),
            'time_since_last_event': self._time_since_last(event),
        }
        
        return pd.DataFrame([features])
    
    def _extract_hour(self, event):
        return pd.to_datetime(event['timestamp']).hour
    
    def _is_night(self, event):
        hour = self._extract_hour(event)
        return hour >= 0 and hour < 6
    
    def _user_failures(self, event, hours):
        # Query Redis for rolling count
        key = f"failures:{event['actor']['user']}:{hours}h"
        return int(self.redis.get(key) or 0)
    
    def _geo_distance(self, event):
        # Calculate distance from user's usual location
        # Simplified - would use actual geo calculation
        return 0.0
    
    def _is_tor(self, event):
        # Check against Tor exit node list
        return False
    
    def _is_privileged(self, event):
        return event['actor']['user'] in ['root', 'admin', 'Administrator']
```

---

# 5. ML/AI System

## 5.1 Model Architecture

```
                    ┌─────────────────────────────┐
                    │      MODEL ENSEMBLE         │
                    └─────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    ANOMALY      │   │   CLASSIFIER    │   │   RULE-BASED    │
│   DETECTOR      │   │   (XGBoost)     │   │    ENGINE       │
│ (Isolation      │   │                 │   │                 │
│  Forest)        │   │ Classes:        │   │ Deterministic:  │
│                 │   │ • Normal        │   │ • Known IOCs    │
│ Unsupervised    │   │ • Brute Force   │   │ • Signatures    │
│ Score: 0.0-1.0  │   │ • Credential    │   │ • Thresholds    │
│                 │   │ • Lateral Move  │   │                 │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │       FUSION LAYER          │
                    │                             │
                    │ final_score = weighted_avg( │
                    │   anomaly * 0.3,            │
                    │   classifier * 0.5,         │
                    │   rules * 0.2               │
                    │ )                           │
                    └─────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │         OUTPUT              │
                    │ {                           │
                    │   "threat_score": 0.87,     │
                    │   "severity": "HIGH",       │
                    │   "attack_type": "brute_f", │
                    │   "confidence": 0.92        │
                    │ }                           │
                    └─────────────────────────────┘
```

Los pesos `0.3 / 0.5 / 0.2` son los del código
([`ml/models/fusion.py`](./ml/models/fusion.py)) y **los mismos que usa el track de flujos**
([`ml/flow/fusion.py`](./ml/flow/fusion.py)), que combina las dos etapas de su cascada con el
mismo reparto. `FusionLayer` normaliza los pesos si no suman 1 y lo registra con
`fusion_weights_not_normalized`.

## 5.2 Model Selection Rationale

| Model | Purpose | Why |
|-------|---------|-----|
| **Isolation Forest** | Anomaly detection | No labels needed, fast, handles high dimensions |
| **XGBoost** | Classification | Handles imbalance, interpretable, production-proven |
| **Rule Engine** | Known patterns | Zero false negatives for critical signatures |

**Why NOT Deep Learning — evidencia medida, no preferencia**

El track de flujos (§5.7) permitió contrastar las dos opciones sobre el **mismo** conjunto de test:
1.000.001 flujos reales de CICIDS2017, test de 150.001, con Stage 1 idéntico en ambas cascadas.
Stage 2 es la única variable: XGBoost contra una red neuronal (MLP con BatchNorm, Adam, focal loss
y dropout).

| Stage 2 | Accuracy | F1-macro | F1-weighted |
|---------|----------|----------|-------------|
| **XGBoost** | 0.9936 | **0.9516** | 0.9936 |
| MLP | 0.9696 | **0.7508** | 0.9747 |

Los 20 puntos de F1-macro los pierde la red **en las clases minoritarias**, que son justamente las
que importan: Botnet 0.1588 contra 0.7616, WebAttack 0.3547 contra 0.9615. El F1-weighted casi
empata (0.9747 vs 0.9936) porque lo domina BENIGN — es la métrica que haría parecer competitiva a
la red y por eso no es la que decide.

> Ambas cifras son **cotas optimistas**: proceden del protocolo con split aleatorio, que filtra
> casi-duplicados de la misma ráfaga entre train y test. Eso **no invalida la comparación** —
> ambos modelos pagan el mismo sesgo sobre el mismo split—, pero ninguno de los dos números estima
> el rendimiento sobre tráfico no visto. Con partición temporal, XGBoost cae a 0.2809 de F1-macro.
> Detalle completo en [docs/ml/README.md](./docs/ml/README.md#resultados).

A la evidencia se suman las razones operativas, que no cambian:
- Security requires explainability (XGBoost da importancia de features directamente)
- Lower operational overhead
- Easier debugging in incident response

La red neuronal **se conserva** en `ml/flow/models/neural_network.py` por valor académico
(implementación en NumPy puro), fuera de la ruta de serving y fuera de la imagen de producción.

## 5.3 Anomaly Detector

```python
# ml/models/anomaly_detector.py

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import numpy as np
import pandas as pd

class AnomalyDetector:
    """
    Isolation Forest based anomaly detector.
    Learns "normal" patterns, flags deviations.
    """
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', IsolationForest(
                n_estimators=200,
                max_samples=0.8,
                contamination=0.01,  # Expect 1% anomalies
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            ))
        ])
        self.feature_names = []
    
    def train(self, X: pd.DataFrame):
        """Train on normal behavior data."""
        self.feature_names = list(X.columns)
        
        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": 200,
                "contamination": 0.01,
                "n_features": len(self.feature_names)
            })
            
            self.pipeline.fit(X)
            
            # Baseline metrics
            scores = self.pipeline.decision_function(X)
            mlflow.log_metrics({
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores))
            })
            
            mlflow.sklearn.log_model(
                self.pipeline,
                "model",
                registered_model_name="sentinel-anomaly-detector"
            )
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Predict anomaly score."""
        raw_score = self.pipeline.decision_function(X)[0]
        
        # Convert to 0-1 (higher = more anomalous)
        anomaly_score = 1 / (1 + np.exp(raw_score))
        
        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": anomaly_score > 0.5,
            "raw_score": float(raw_score)
        }
```

## 5.4 Attack Classifier

```python
# ml/models/classifier.py

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import mlflow
import numpy as np
import pandas as pd

class AttackClassifier:
    """
    XGBoost classifier for attack type categorization.
    """
    
    CLASSES = {
        0: "normal",
        1: "brute_force",
        2: "credential_stuffing",
        3: "privilege_escalation",
        4: "lateral_movement",
        5: "data_exfiltration",
        6: "reconnaissance"
    }
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,  # Handle imbalance
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train with cross-validation."""
        with mlflow.start_run():
            mlflow.log_params(self.model.get_params())
            
            # 5-fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                self.model.fit(
                    X.iloc[train_idx], y.iloc[train_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    early_stopping_rounds=20,
                    verbose=False
                )
                score = self.model.score(X.iloc[val_idx], y.iloc[val_idx])
                scores.append(score)
                mlflow.log_metric(f"cv_fold_{fold}", score)
            
            # Final train
            self.model.fit(X, y)
            
            mlflow.log_metrics({
                "cv_mean": np.mean(scores),
                "cv_std": np.std(scores)
            })
            
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name="sentinel-attack-classifier"
            )
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Predict attack type."""
        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        
        return {
            "attack_type": self.CLASSES[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": {
                self.CLASSES[i]: float(p) 
                for i, p in enumerate(probs)
            }
        }
```

## 5.5 MLflow Configuration

```yaml
# ml/mlflow/config.yaml

tracking:
  uri: "gs://sentinel-mlflow/tracking"
  artifact_location: "gs://sentinel-mlflow/artifacts"

experiments:
  - name: "anomaly-detection"
    tags:
      model_type: "unsupervised"
  - name: "attack-classification"
    tags:
      model_type: "supervised"

registry:
  models:
    - sentinel-anomaly-detector
    - sentinel-attack-classifier
  stages:
    - None
    - Staging
    - Production
    - Archived
```

## 5.6 Training Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    DATA     │───►│   FEATURE   │───►│  TRAINING   │───►│   MODEL     │
│   LOADING   │    │  PIPELINE   │    │             │    │  REGISTRY   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
 GCS bucket         Transform +        MLflow            Version +
 raw/ →             Validate          Experiment         Stage
 processed/                           Tracking          (Staging)

                    ┌─────────────────────────────────────────┐
                    │           VALIDATION GATE               │
                    │                                         │
                    │  • Accuracy > 0.95                      │
                    │  • F1-Score > 0.90                      │
                    │  • Latency < 50ms p99                   │
                    │                                         │
                    │  ALL pass → Promote to Production       │
                    │  ANY fail → Alert + Manual Review       │
                    └─────────────────────────────────────────┘
```

> **Nota de honestidad:** el VALIDATION GATE está **documentado pero no implementado**. Ningún
> code path de `ml/training/train.py` compara las métricas contra esos umbrales ni bloquea la
> promoción. Queda como mejora identificada (~25 líneas), no como comportamiento actual.

## 5.7 Flow-Based Track (CICIDS2017)

Sentinel detecta intrusiones sobre **dos modalidades de dato en paralelo**. Todo lo anterior en §5
describe el track de **eventos de log**. El track de **flujos de red** es independiente: otra
ontología, otra taxonomía, otros artefactos y otros endpoints.

| | Track de eventos | Track de flujos |
|---|---|---|
| Unidad | Evento de log | Flujo bidireccional (CICFlowMeter) |
| Esquema | `data/schemas/event.json` | `data/schemas/flow.json` |
| Features | 25 derivadas | 78 columnas CICFlowMeter |
| Taxonomía | `normal`, `brute_force`, … (7) | `BENIGN`, `Botnet`, `BruteForce`, `DDoS`, `DoS`, `Scan`, `WebAttack` |
| Topología | **Ensemble** (3 detectores en paralelo → fusión) | **Cascada** (Stage 1 → Stage 2 → fusión) |
| Código | `ml/models/`, `ml/features/` | `ml/flow/` |
| Entrenamiento | `python -m ml.training.train` | `python -m ml.training.train_cicids` |
| Artefactos | `models/` | `models/flow/` |
| Datos | Generador sintético | CICIDS2017 (2,83 M de flujos reales) |

### Arquitectura de cascada

```
flujo (78 features)
      │
      ▼
┌──────────────────────────┐
│ STAGE 1                  │  Isolation Forest, no supervisado.
│ FlowAnomalyDetector      │  Umbral por F-beta (β=2): penaliza los
└──────┬───────────────────┘  ataques no detectados.
       │
  ¿anómalo? ── NO ──►  BENIGN. Stage 2 nunca lo ve.
       │
      SÍ
       ▼
┌──────────────────────────┐
│ STAGE 2                  │  XGBoost multiclase (7 familias).
│ FlowAttackClassifier     │
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ FlowFusionLayer          │  0.3 anomalía + 0.5 clasificación + 0.2 reglas
└──────────────────────────┘
```

A diferencia del ensemble de eventos, **el recall de Stage 1 es un techo duro**: lo que Stage 1
deja pasar como benigno no llega a Stage 2 y no tiene segunda oportunidad. De ahí el umbral con
β=2 en vez de F1.

### Taxonomías separadas a propósito

Las dos listas tienen 7 entradas y `brute_force` se parece a `BruteForce`, pero **unificarlas sería
un error**. `AttackClassifier._encode_labels` mapea etiqueta desconocida → `normal` **en silencio**:
alimentar el clasificador de eventos con etiquetas CICIDS produce un modelo que predice `normal`
para todo, entrena sin lanzar una excepción y pasa cualquier smoke que mire el exit code. Módulos
y clases separados lo hacen imposible por construcción.

### Partición temporal

El TFA de origen usaba un `train_test_split` estratificado **por fila** sobre una captura continua,
lo que filtra casi-duplicados de la misma ráfaga entre train y test. El track de sentinel corta
**70/15/15 cronológicamente dentro de cada bloque `(día de captura, clase)`** (`--split temporal`,
el default). El impacto medido, mismo comando salvo el `--split`:

| Split | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| `random` (protocolo del TFA) | 0.9934 | **0.9166** | 0.9935 |
| `temporal` (default) | 0.8783 | **0.2809** | 0.8439 |

**Esa caída no es una regresión: es la magnitud de la fuga que el protocolo anterior ocultaba.**

### Contrato de features

`FLOW_FEATURE_NAMES` congela las 78 columnas en orden canónico. El extractor reordena **por
nombre**, y `CascadeInferenceService.load()` valida la lista persistida en el artefacto contra esa
constante, fallando con `FeatureContractError` si difiere en longitud, contenido **u orden**. Es la
defensa contra el fallo más caro del diseño: un desajuste de orden entre entrenamiento e inferencia
**no lanza ninguna excepción**, solo produce predicciones basura con forma perfectamente válida.

### Degradación independiente

- **`/ready` lo decide solo el track de eventos.** Los checks `flow_detector` y `flow_classifier`
  se reportan pero **no cambian el código de estado**: un modelo de flujo ausente no puede sacar el
  pod de servicio y llevarse por delante los endpoints de eventos.
- Sin cascada cargada, las rutas de flujo devuelven **503** y las de eventos siguen en **200**.
- `PREDICTION_SCORE` y `ATTACK_TYPE_COUNT` ganan una etiqueta **`track`** (`event`|`flow`); no hay
  colectores duplicados y las queries existentes siguen funcionando porque agregan con `sum(...)`.

Documentación completa del track: [docs/ml/CICIDS2017-TRACK.md](./docs/ml/CICIDS2017-TRACK.md).

---

# 6. API Layer

## 6.1 API Endpoints

| Endpoint | Method | Description | Auth | Rate Limit |
|----------|--------|-------------|------|------------|
| `/v1/predict` | POST | Single event analysis | JWT | 100/min |
| `/v1/analyze/batch` | POST | Batch analysis (max 100) | JWT | 10/min |
| `/v1/predict/flow` | POST | Single network flow analysis (§5.7) | JWT | 100/min |
| `/v1/analyze/flow/batch` | POST | Batch flow analysis (max 100) | JWT | 10/min |
| `/health` | GET | Health check | None | 1000/min |
| `/ready` | GET | Readiness check | None | 1000/min |
| `/metrics` | GET | Prometheus metrics | None | 60/min |

Las dos rutas de flujo devuelven **503** si no hay cascada cargada, de forma **independiente** de
las rutas de evento. `/v1/analyze/flow/batch` no itera predicciones individuales: pasa la matriz
entera a `predict_batch`, así que cien flujos son dos invocaciones de modelo, no doscientas.

## 6.2 Request/Response Schema

### Predict Request
```json
{
  "timestamp": "2024-01-15T14:32:18.123Z",
  "source": {
    "type": "auth",
    "hostname": "web-server-01",
    "ip": "10.0.1.45"
  },
  "actor": {
    "user": "admin",
    "source_ip": "203.0.113.45"
  },
  "action": {
    "type": "authentication",
    "subtype": "login_failed",
    "result": "failure"
  }
}
```

### Predict Response
```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "threat_score": 0.87,
  "severity": "HIGH",
  "attack_type": "brute_force",
  "confidence": 0.92,
  "is_anomaly": true,
  "explanation": [
    "47 failed logins from this IP in last hour",
    "Source IP is from high-risk country",
    "Activity outside normal business hours"
  ],
  "recommended_actions": [
    "Block source IP temporarily",
    "Notify security team",
    "Enable MFA for target user"
  ]
}
```

## 6.3 FastAPI Implementation

```python
# api/main.py

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address
import structlog
import time

app = FastAPI(title="Sentinel Security API", version="1.0.0")
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)
logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total requests',
    ['endpoint', 'method', 'status']
)
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency',
    ['endpoint']
)

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start)
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    return response

# Auth dependency
async def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
@app.post("/v1/predict")
@limiter.limit("100/minute")
async def predict(request: Request, event: SecurityEvent, token: dict = Depends(verify_token)):
    """Analyze security event and return threat assessment."""
    logger.info("prediction_request", event_type=event.source.type)
    
    try:
        # Extract features
        features = feature_pipeline.transform(event)
        
        # Get predictions
        anomaly = anomaly_detector.predict(features)
        classification = classifier.predict(features)
        rules = rule_engine.evaluate(event)
        
        # Combine
        result = fusion.combine(anomaly, classification, rules)
        
        # Alert if severe
        if result.severity in ["HIGH", "CRITICAL"]:
            await alerter.send(result)
        
        return result
        
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    checks = {
        "model_loaded": model_server.is_ready(),
        "feature_pipeline": feature_pipeline.is_ready()
    }
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", "checks": checks}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

## 6.4 Dockerfile

```dockerfile
# api/Dockerfile

FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM gcr.io/distroless/python3-debian12

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# 7. Kubernetes Platform

## 7.1 Namespace Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  sentinel-platform                                               │
│  ├── ingestion-service                                          │
│  ├── feature-pipeline                                           │
│  └── configs                                                    │
│  Network Policy: Allow from sentinel-ml                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  sentinel-ml                                                     │
│  ├── prediction-api                                             │
│  ├── model-server                                               │
│  └── training-jobs (CronJobs)                                   │
│  Network Policy: Allow from Ingress, to Storage                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  sentinel-observability                                          │
│  ├── prometheus                                                 │
│  ├── grafana                                                    │
│  └── alertmanager                                               │
│  Network Policy: Allow scraping from all                        │
└─────────────────────────────────────────────────────────────────┘
```

## 7.2 Prediction API Deployment

```yaml
# k8s/base/ml/prediction-api/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-api
  namespace: sentinel-ml
spec:
  replicas: 2
  selector:
    matchLabels:
      app: prediction-api
  template:
    metadata:
      labels:
        app: prediction-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      serviceAccountName: prediction-api-sa
      
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      
      containers:
        - name: api
          image: gcr.io/PROJECT_ID/sentinel-api:v1.0.0
          
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          
          ports:
            - containerPort: 8000
          
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                configMapKeyRef:
                  name: ml-config
                  key: mlflow-uri
          
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      
      volumes:
        - name: tmp
          emptyDir: {}
      
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: prediction-api
```

## 7.3 Horizontal Pod Autoscaler

```yaml
# k8s/base/ml/prediction-api/hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prediction-api-hpa
  namespace: sentinel-ml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
```

## 7.4 Network Policy

```yaml
# k8s/base/ml/network-policy.yaml

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-namespace-policy
  namespace: sentinel-ml
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # From ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
    # From observability (Prometheus)
    - from:
        - namespaceSelector:
            matchLabels:
              name: sentinel-observability
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # GCP APIs
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

## 7.5 Service Account & RBAC

```yaml
# k8s/base/rbac/prediction-api.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
  name: prediction-api-sa
  namespace: sentinel-ml
  annotations:
    iam.gke.io/gcp-service-account: sentinel-api@PROJECT_ID.iam.gserviceaccount.com

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prediction-api-role
  namespace: sentinel-ml
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get"]
    resourceNames: ["api-secrets"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prediction-api-binding
  namespace: sentinel-ml
subjects:
  - kind: ServiceAccount
    name: prediction-api-sa
roleRef:
  kind: Role
  name: prediction-api-role
  apiGroup: rbac.authorization.k8s.io
```

---

# 8. CI/CD Pipeline

## 8.1 Pipeline Overview

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  CODE   │──►│  TEST   │──►│  SCAN   │──►│  BUILD  │──►│ DEPLOY  │
│  PUSH   │   │         │   │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
                  │             │             │             │
                  ▼             ▼             ▼             ▼
              Unit Tests    SAST         Docker        Staging
              Lint          Semgrep      Build         Deploy
              Type Check    pip-audit    Push GCR      Smoke Test
                            Trivy                      Prod Deploy

              ┌────────────────────────────────────────────────┐
              │              SECURITY GATE                     │
              │                                                │
              │  If ANY security check fails → BLOCK DEPLOY    │
              └────────────────────────────────────────────────┘
```

## 8.2 CI Workflow

```yaml
# .github/workflows/ci.yaml

name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PROJECT_ID: sentinel-platform
  REGION: europe-west1

jobs:
  test:
    name: Test & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Lint (Ruff)
        run: |
          ruff check .
          ruff format --check .
      
      - name: Type check (mypy)
        run: mypy api/ ml/
      
      - name: Unit tests
        run: |
          pytest tests/unit \
            --cov=api --cov=ml \
            --cov-report=xml \
            --cov-fail-under=80

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: SAST (Semgrep)
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/python p/security-audit p/owasp-top-ten
      
      - name: Dependency scan (pip-audit)
        run: |
          pip install pip-audit
          pip-audit -r requirements.txt
      
      - name: Secrets scan (Gitleaks)
        uses: gitleaks/gitleaks-action@v2

  build:
    name: Build & Scan Image
    runs-on: ubuntu-latest
    needs: [test, security]
    steps:
      - uses: actions/checkout@v4
      
      - uses: docker/setup-buildx-action@v3
      
      - name: Auth to GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SA }}
      
      - name: Configure Docker
        run: gcloud auth configure-docker gcr.io
      
      - name: Build image
        uses: docker/build-push-action@v5
        with:
          context: ./api
          push: false
          load: true
          tags: gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }}
      
      - name: Scan image (Trivy)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }}
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
      
      - name: Push image
        run: docker push gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }}

  deploy-staging:
    name: Deploy Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - uses: google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: sentinel-cluster
          location: ${{ env.REGION }}
      
      - name: Deploy
        run: |
          kubectl set image deployment/prediction-api \
            api=gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }} \
            -n sentinel-ml-staging
          kubectl rollout status deployment/prediction-api -n sentinel-ml-staging
      
      - name: Smoke tests
        run: ./scripts/smoke-tests.sh staging

  deploy-prod:
    name: Deploy Production
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - uses: google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: sentinel-cluster
          location: ${{ env.REGION }}
      
      - name: Deploy canary (10%)
        run: |
          kubectl set image deployment/prediction-api-canary \
            api=gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }} \
            -n sentinel-ml
      
      - name: Monitor canary
        run: ./scripts/monitor-canary.sh 300
      
      - name: Full rollout
        run: |
          kubectl set image deployment/prediction-api \
            api=gcr.io/${{ env.PROJECT_ID }}/sentinel-api:${{ github.sha }} \
            -n sentinel-ml
          kubectl rollout status deployment/prediction-api -n sentinel-ml
```

## 8.3 Terraform Workflow

```yaml
# .github/workflows/terraform.yaml

name: Terraform

on:
  push:
    branches: [main]
    paths: ['infra/terraform/**']
  pull_request:
    paths: ['infra/terraform/**']

jobs:
  terraform:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: infra/terraform/environments/prod
    steps:
      - uses: actions/checkout@v4
      
      - uses: hashicorp/setup-terraform@v3
      
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SA }}
      
      - name: Init
        run: terraform init
      
      - name: Format check
        run: terraform fmt -check -recursive
      
      - name: Validate
        run: terraform validate
      
      - name: Security scan (tfsec)
        uses: aquasecurity/tfsec-action@v1.0.3
      
      - name: Plan
        run: terraform plan -out=tfplan
      
      - name: Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve tfplan
```

---

# 9. Observability Stack

## 9.1 Metrics (Prometheus)

### Key Metrics

```yaml
# API Performance
- sentinel:api_request_rate_5m
- sentinel:api_error_rate_5m
- sentinel:api_latency_p99
- sentinel:api_latency_p95

# ML Model
- sentinel:prediction_score_avg
- sentinel:high_threat_rate_1h
- sentinel:model_latency_p99

# Infrastructure
- container_cpu_usage_seconds_total
- container_memory_usage_bytes
- kube_pod_status_phase
```

### Alert Rules

```yaml
# observability/prometheus/rules.yaml

groups:
  - name: sentinel-alerts
    rules:
      - alert: HighErrorRate
        expr: sentinel:api_error_rate_5m > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: HighLatency
        expr: sentinel:api_latency_p99 > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
      
      - alert: AnomalySpike
        expr: sentinel:high_threat_rate_1h > 0.2
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Unusual spike in threats"
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total{namespace=~"sentinel-.*"}[15m]) > 0
        for: 5m
        labels:
          severity: critical
```

## 9.2 Alertmanager

```yaml
# observability/alertmanager/config.yaml

global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  routes:
    - match:
        severity: critical
      receiver: 'critical'
    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#sentinel-alerts'
  
  - name: 'critical'
    slack_configs:
      - channel: '#sentinel-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_KEY}'
  
  - name: 'warning'
    slack_configs:
      - channel: '#sentinel-alerts'
```

## 9.3 Grafana Dashboards

### Dashboard: Sentinel Overview
- Request Rate by endpoint
- Latency Distribution (heatmap)
- Threat Score Distribution
- Attack Types (pie chart)
- Error Rate (gauge)
- Model Performance (gauge)

---

# 10. Security Architecture

## 10.1 Defense in Depth

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: PERIMETER                                               │
│ • Cloud Armor (WAF, DDoS)                                       │
│ • OWASP rules, Geo-blocking                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ LAYER 2: NETWORK                                                 │
│ • Private VPC, no public IPs                                    │
│ • Network Policies, default deny                                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ LAYER 3: IDENTITY                                                │
│ • IAM least privilege                                           │
│ • Workload Identity (no keys)                                   │
│ • JWT authentication                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ LAYER 4: COMPUTE                                                 │
│ • Distroless images                                             │
│ • Non-root, read-only filesystem                                │
│ • Pod Security Standards                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ LAYER 5: DATA                                                    │
│ • Encryption at rest (CMEK)                                     │
│ • TLS 1.3 in transit                                            │
│ • Secret Manager                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ LAYER 6: APPLICATION                                             │
│ • Input validation (Pydantic)                                   │
│ • Rate limiting                                                 │
│ • No sensitive data in logs                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 10.2 Security Checklist

### Infrastructure
- [ ] Private GKE cluster
- [ ] Cloud NAT for egress
- [ ] VPC firewall default deny
- [ ] Cloud Armor WAF active
- [ ] Binary Authorization enabled

### Kubernetes
- [ ] Pod Security Standards enforced
- [ ] Network Policies deployed
- [ ] RBAC configured (least privilege)
- [ ] Secrets from Secret Manager
- [ ] No privileged containers

### Application
- [ ] JWT authentication
- [ ] Input validation
- [ ] Rate limiting
- [ ] Structured logging (no secrets)
- [ ] HTTPS only

### CI/CD
- [ ] SAST scanning (Semgrep)
- [ ] Dependency scanning (pip-audit)
- [ ] Container scanning (Trivy)
- [ ] No secrets in code
- [ ] Signed images

---

# 11. Implementation Roadmap

## Phase 1: Foundation (Weeks 1-2)

### Week 1: Infrastructure
- [ ] Create GCP project
- [ ] Set up Terraform backend (GCS)
- [ ] Deploy VPC module
- [ ] Deploy GKE cluster
- [ ] Configure IAM & Workload Identity

### Week 2: Base Platform
- [ ] Deploy storage buckets
- [ ] Set up Secret Manager
- [ ] Configure namespaces
- [ ] Deploy RBAC
- [ ] Set up CI/CD skeleton

**Deliverable:** Working GKE cluster with networking

## Phase 2: ML Pipeline (Weeks 3-4)

### Week 3: Data & Features
- [ ] Create log schema
- [ ] Build synthetic data generator
- [ ] Implement feature pipeline
- [ ] Set up MLflow tracking

### Week 4: Models
- [ ] Train anomaly detector
- [ ] Train attack classifier
- [ ] Implement fusion layer
- [ ] Register models in MLflow

**Deliverable:** Trained models in registry

## Phase 3: API & Serving (Weeks 5-6)

### Week 5: API Development
- [ ] Implement FastAPI endpoints
- [ ] Add authentication (JWT)
- [ ] Implement rate limiting
- [ ] Add metrics instrumentation
- [ ] Write Dockerfile

### Week 6: Kubernetes Deployment
- [ ] Create K8s manifests
- [ ] Configure HPA
- [ ] Set up Network Policies
- [ ] Deploy to GKE
- [ ] Test end-to-end

**Deliverable:** Working API in Kubernetes

## Phase 4: Production Readiness (Weeks 7-8)

### Week 7: CI/CD & Security
- [ ] Complete GitHub Actions workflows
- [ ] Add SAST/DAST scanning
- [ ] Configure security gates
- [ ] Set up staging environment

### Week 8: Observability
- [ ] Deploy Prometheus
- [ ] Deploy Grafana
- [ ] Create dashboards
- [ ] Configure alerts
- [ ] Write runbooks

**Deliverable:** Production-ready platform

## Future Enhancements
- Policy as Code (OPA)
- Model drift detection
- Feature store (Feast)
- Chaos engineering
- Multi-cluster deployment

---

# 12. Repository Structure

```
sentinel-platform/
│
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       ├── terraform.yaml
│       └── ml-training.yaml
│
├── infra/
│   └── terraform/
│       ├── modules/
│       │   ├── vpc/
│       │   ├── gke/
│       │   ├── storage/
│       │   ├── iam/
│       │   └── security/
│       ├── environments/
│       │   ├── dev/
│       │   ├── staging/
│       │   └── prod/
│       └── backend.tf
│
├── k8s/
│   ├── base/
│   │   ├── namespaces/
│   │   ├── platform/
│   │   ├── ml/
│   │   ├── observability/
│   │   └── rbac/
│   └── overlays/
│       ├── dev/
│       ├── staging/
│       └── prod/
│
├── api/
│   ├── main.py
│   ├── models/
│   ├── middleware/
│   ├── security/
│   ├── Dockerfile
│   └── requirements.txt
│
├── ml/
│   ├── features/
│   │   └── pipeline.py
│   ├── models/
│   │   ├── anomaly_detector.py
│   │   └── classifier.py
│   ├── training/
│   │   └── train.py
│   └── serving/
│       └── server.py
│
├── data/
│   ├── schemas/
│   └── generators/
│       └── synthetic_logs.py
│
├── observability/
│   ├── prometheus/
│   │   └── rules.yaml
│   ├── grafana/
│   │   └── dashboards/
│   └── alertmanager/
│       └── config.yaml
│
├── scripts/
│   ├── setup.sh
│   ├── smoke-tests.sh
│   └── monitor-canary.sh
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── architecture.md
│   ├── runbooks/
│   └── adr/
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

# 13. Getting Started

## 13.1 Prerequisites

```bash
# Tools required
gcloud --version          # Google Cloud SDK
terraform --version       # >= 1.6.0
kubectl version          # Kubernetes CLI
python --version         # >= 3.11
docker --version         # Docker
```

## 13.2 Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/guillesanper/sentinel-platform.git
cd sentinel-platform

# 2. Authenticate to GCP
gcloud auth login
gcloud config set project sentinel-platform

# 3. Enable required APIs
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  cloudresourcemanager.googleapis.com

# 4. Create Terraform backend bucket
gsutil mb -l europe-west1 gs://sentinel-terraform-state

# 5. Initialize Terraform
cd infra/terraform/environments/dev
terraform init
terraform plan
terraform apply

# 6. Get GKE credentials
gcloud container clusters get-credentials sentinel-cluster \
  --region europe-west1

# 7. Deploy Kubernetes resources
kubectl apply -k k8s/overlays/dev

# 8. Verify deployment
kubectl get pods -n sentinel-ml
```

## 13.3 Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Run tests
pytest tests/unit -v

# 4. Run API locally
cd api
uvicorn main:app --reload --port 8000

# 5. Test endpoint
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d @sample_event.json
```

## 13.4 Training Models

```bash
# 1. Generate synthetic data
python data/generators/synthetic_logs.py \
  --output gs://sentinel-data-raw/synthetic \
  --count 100000

# 2. Run feature pipeline
python ml/features/pipeline.py \
  --input gs://sentinel-data-raw/synthetic \
  --output gs://sentinel-data-processed/features

# 3. Train models
python ml/training/train.py \
  --features gs://sentinel-data-processed/features \
  --experiment sentinel-training-v1

# 4. Promote to production
mlflow models transition \
  --name sentinel-anomaly-detector \
  --version 1 \
  --stage Production
```

---

# Appendix A: Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cloud | GCP | GKE maturity, managed Prometheus |
| IaC | Terraform | Industry standard, modularity |
| Container Runtime | GKE | Production-grade, Workload Identity |
| API Framework | FastAPI | Async, OpenAPI, performance |
| ML Models | sklearn + XGBoost | Interpretable, production-proven |
| MLOps | MLflow | Open source, GCS backend |
| Observability | Prometheus + Grafana | Cloud-native standard |
| CI/CD | GitHub Actions | Native integration |

---

# Appendix B: Glossary

| Term | Definition |
|------|------------|
| **IDS** | Intrusion Detection System |
| **IaC** | Infrastructure as Code |
| **MLOps** | Machine Learning Operations |
| **DevSecOps** | Development, Security, Operations |
| **GKE** | Google Kubernetes Engine |
| **WAF** | Web Application Firewall |
| **SAST** | Static Application Security Testing |
| **HPA** | Horizontal Pod Autoscaler |
| **RBAC** | Role-Based Access Control |

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Author:** [Tu nombre]