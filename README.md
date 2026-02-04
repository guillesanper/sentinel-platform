# Sentinel Platform

[![CI](https://github.com/yourusername/sentinel-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/yourusername/sentinel-platform/actions/workflows/ci.yaml)
[![Security](https://github.com/yourusername/sentinel-platform/actions/workflows/security.yaml/badge.svg)](https://github.com/yourusername/sentinel-platform/actions/workflows/security.yaml)

> Cloud-Native Intrusion Detection Platform powered by Machine Learning

## 🎯 Overview

**Sentinel** es una plataforma de detección de intrusiones que analiza logs de infraestructura en tiempo real para identificar:

- 🔐 **Brute Force** - Intentos repetidos de autenticación
- 🎭 **Credential Stuffing** - Uso de credenciales robadas
- ⬆️ **Privilege Escalation** - Escalada de privilegios no autorizada
- 🔀 **Lateral Movement** - Movimiento lateral en la red
- 📤 **Data Exfiltration** - Exfiltración de datos
- 🔍 **Reconnaissance** - Actividades de reconocimiento

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
│   ├── features/             # Feature engineering
│   ├── models/               # Model definitions
│   └── training/             # Training pipelines
├── data/                     # Data schemas and generators
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
git clone https://github.com/yourusername/sentinel-platform.git
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

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/predict` | POST | Analyze single event |
| `/v1/analyze/batch` | POST | Analyze batch (max 100) |

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
- [Technical Design Document](./docs/SENTINEL_TECHNICAL_DESIGN.md)
- [Architecture Decisions](./docs/adr/)
- [Runbooks](./docs/runbooks/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.