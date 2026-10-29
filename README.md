# Sentinel Platform

[![CI](https://github.com/guillesanper/sentinel-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/guillesanper/sentinel-platform/actions/workflows/ci.yaml)

> Cloud-Native Intrusion Detection Platform powered by Machine Learning

## 🎯 Overview

**Sentinel** es una plataforma de detección de intrusiones que analiza **dos modalidades de dato en
paralelo**: eventos de log de infraestructura y flujos de red.

**Track de eventos** — sobre logs de infraestructura en tiempo real:

- 🔐 **Brute Force** - Intentos repetidos de autenticación
- 🎭 **Credential Stuffing** - Uso de credenciales robadas
- ⬆️ **Privilege Escalation** - Escalada de privilegios no autorizada
- 🔀 **Lateral Movement** - Movimiento lateral en la red
- 📤 **Data Exfiltration** - Exfiltración de datos
- 🔍 **Reconnaissance** - Actividades de reconocimiento

**Track de flujos** — sobre flujos de red bidireccionales (features CICFlowMeter, 78 columnas),
con una cascada Isolation Forest → XGBoost entrenada sobre CICIDS2017:

- 💥 **DoS / DDoS** · 🤖 **Botnet** · 🔓 **BruteForce** · 🔎 **Scan** · 🌐 **WebAttack**

Los dos tracks tienen taxonomías, modelos y endpoints **separados a propósito** — ver
[docs/ml/](./docs/ml/README.md).

## 📊 Key Results (flow track, CICIDS2017)

| Split | F1-macro | Stage 1 recall / FPR |
|---|---|---|
| Random (row-level) | 0.9166 | 0.976 / 0.625 |
| **Temporal** (default in this repo) | **0.2809** | 0.626 / 0.034 |

The drop is the size of the leakage that a row-level random split hides: near-duplicate flows from
the same attack burst land in both train and test. Sentinel evaluates with a temporal split by
default and publishes both numbers with their provenance — full analysis in
[docs/ml/README.md](./docs/ml/README.md#resultados).

## 🏗️ Architecture

```
                              ┌─────────────────┐
                              │   Cloud Armor   │
                              │   (WAF/DDoS)    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Cloud Load    │
                              │   Balancer      │
                              └────────┬────────┘
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                           GKE PRIVATE CLUSTER                                 │
│                                                                               │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │    PLATFORM     │    │       ML        │    │  OBSERVABILITY  │         │
│   │   NAMESPACE     │    │   NAMESPACE     │    │   NAMESPACE     │         │
│   │                 │    │                 │    │                 │         │
│   │ • Ingestion     │◄──►│ • Prediction    │    │ • Prometheus    │         │
│   │ • Features      │    │   API           │    │ • Grafana       │         │
│   │                 │    │ • Model Server  │    │ • Alertmanager  │         │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Cloud** | Google Cloud Platform |
| **IaC** | Terraform |
| **Orchestration** | Kubernetes (GKE) |
| **API** | FastAPI + Uvicorn |
| **ML** | scikit-learn, XGBoost, MLflow |
| **Observability** | Prometheus, Grafana, Alertmanager |
| **CI/CD** | GitHub Actions |
| **Security** | Trivy, Semgrep, pip-audit |

## 📁 Project Structure

```
sentinel-platform/
├── infra/terraform/          # Infrastructure as Code
│   ├── modules/              # Reusable Terraform modules
│   └── environments/         # Environment configurations
├── k8s/                      # Kubernetes manifests
│   ├── base/                 # Base configurations
│   └── overlays/             # Environment overlays
├── api/                      # FastAPI application
├── ml/                       # Machine Learning code
│   ├── features/             # Feature engineering (track de eventos)
│   ├── models/               # Model definitions (track de eventos)
│   ├── flow/                 # Track de flujos: loader, cascada, serving
│   ├── evaluation/           # Métricas y figuras (compartido por ambos tracks)
│   └── training/             # Training pipelines (train.py, train_cicids.py)
├── data/                     # Data schemas and generators
├── docs/ml/                  # Documentación de ML + material del TFA CICIDS2017
├── observability/            # Monitoring configuration
├── scripts/                  # Utility scripts
└── tests/                    # Test suites
```

## 🚀 Quick Start

### Prerequisites

```bash
# Required tools
gcloud --version          # Google Cloud SDK
terraform --version       # >= 1.6.0
kubectl version           # Kubernetes CLI
python --version          # >= 3.11
docker --version          # Docker
```

### 1. Clone & Setup

```bash
git clone https://github.com/guillesanper/sentinel-platform.git
cd sentinel-platform

# Create virtual environment
python -3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. GCP Setup

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com
```

### 3. Deploy Infrastructure

```bash
cd infra/terraform/environments/dev
terraform init
terraform plan
terraform apply
```

### 4. Deploy to Kubernetes

```bash
# Get credentials
gcloud container clusters get-credentials sentinel-cluster --region europe-west1

# Configure secrets in GCP Secret Manager
export PROJECT_ID=your-project-id
echo -n "your-jwt-secret" | gcloud secrets versions add jwt-secret-dev --data-file=-
echo -n "your-api-key" | gcloud secrets versions add api-key-dev --data-file=-

# Update PROJECT_ID in overlays
sed -i "s/PROJECT_ID/${PROJECT_ID}/g" k8s/overlays/dev/kustomization.yaml

# Deploy
./scripts/deploy-k8s.sh dev
```

> **Nota**: Los secrets se gestionan automáticamente mediante External Secrets Operator (ESO),
> que sincroniza desde GCP Secret Manager. Ver [Guia de Deployment](./docs/DEPLOYMENT.md) para mas detalles.

### 5. Run Locally (Development)

```bash
cd api
uvicorn main:app --reload --port 8000

# Test endpoint
curl http://localhost:8000/health
```

### 6. Train Models

```bash
# Track de eventos (generador sintético, sin datos externos)
python -m ml.training.train --n-samples 60000 --attack-ratio 0.2 \
  --threshold-strategy fbeta --beta 2.0 --output-dir models --no-mlflow

# Track de flujos (requiere los CSV de CICIDS2017 y ~8 GiB de RAM)
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 \
  --pipeline xgb --split temporal --skip-tuning --output-dir models/flow
```

Los CSV de CICIDS2017 (844 MB) **no se versionan**: se descargan de
[UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html), distribución `MachineLearningCSV`.
Detalles de opciones y resultados en [docs/ml/CICIDS2017-TRACK.md](./docs/ml/CICIDS2017-TRACK.md).

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/predict` | POST | Analyze single event |
| `/v1/analyze/batch` | POST | Analyze batch of events (max 100) |
| `/v1/predict/flow` | POST | Analyze single network flow |
| `/v1/analyze/flow/batch` | POST | Analyze batch of flows (max 100) |

Las rutas de flujo degradan **independientemente**: si no hay cascada de flujos cargada devuelven
503, mientras las de eventos siguen sirviendo con normalidad.

### Example Request

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "timestamp": "2024-01-15T14:32:18Z",
    "source": {"type": "auth", "hostname": "server-01", "ip": "10.0.1.45"},
    "actor": {"user": "admin", "source_ip": "203.0.113.45"},
    "action": {"type": "authentication", "subtype": "login_failed", "result": "failure"}
  }'
```

### Example Response

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
    "Source IP from high-risk region",
    "Activity outside business hours"
  ]
}
```

### Example Flow Request

Las 78 features son individualmente opcionales (ausente → 0.0) y se reordenan **por nombre**, no
por orden de aparición. Los metadatos van **fuera** de `features` y no entran al modelo:

```bash
curl -X POST http://localhost:8000/v1/predict/flow \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "features": {"Destination Port": 80, "Flow Duration": 1532000, "Total Fwd Packets": 12},
    "src_ip": "192.168.10.5",
    "dst_ip": "10.0.1.20",
    "dst_port": 80,
    "protocol": "TCP"
  }'
```

```json
{
  "flow_id": "550e8400-e29b-41d4-a716-446655440000",
  "threat_score": 0.91,
  "severity": "CRITICAL",
  "attack_type": "DDoS",
  "confidence": 0.88,
  "is_anomaly": true,
  "explanation": [
    "Flow deviates from benign traffic (anomaly score: 0.94)",
    "Traffic pattern matches DDoS (97% confidence)",
    "Very high throughput (42.0 MB/s)"
  ],
  "attack_probabilities": {"BENIGN": 0.01, "DDoS": 0.97, "DoS": 0.02}
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=api --cov=ml --cov-report=html

# Only unit tests
pytest tests/unit -v
```

## 🔒 Security

- Private GKE cluster (no public node IPs)
- Workload Identity (no service account keys)
- Network Policies for pod isolation
- Pod Security Standards enforced
- Container image scanning with Trivy
- SAST with Semgrep
- Dependency scanning with pip-audit

## 📈 Monitoring

Access Grafana dashboards:

```bash
kubectl port-forward svc/grafana 3000:3000 -n sentinel-observability
# Open http://localhost:3000
```

## 📖 Documentation

- [Guia de Deployment](./docs/DEPLOYMENT.md) - Proceso completo de despliegue
- [Guia de Migracion](./docs/MIGRATION_GUIDE.md) - Migracion desde deploy.sh
- [Troubleshooting de Secrets](./docs/SECRETS_TROUBLESHOOTING.md) - Problemas con ESO
- [Modulo Kubernetes Terraform](./infra/terraform/modules/kubernetes/README.md) - Configuracion de ESO
- [Technical Design Document](./sentinel_doc.md) - Diseno tecnico completo (infra, ML, API, k8s, CI/CD)
- [Documentacion de ML](./docs/ml/README.md) - Los dos tracks, resultados y su procedencia
- [Track de flujos (CICIDS2017)](./docs/ml/CICIDS2017-TRACK.md) - Cascada, contrato de features, particion temporal, entrenamiento

> Pendientes de escribir: `docs/adr/` (Architecture Decision Records) y `docs/runbooks/`.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 👤 Author

Guillermo Sánchez — [GitHub](https://github.com/guillesanper) · [Web](https://guillesanper.github.io)
