# Sentinel Platform

[![CI](https://github.com/yourusername/sentinel-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/yourusername/sentinel-platform/actions/workflows/ci.yaml)
[![Security](https://github.com/yourusername/sentinel-platform/actions/workflows/security.yaml/badge.svg)](https://github.com/yourusername/sentinel-platform/actions/workflows/security.yaml)

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

La plataforma **se ejecuta y se prueba entera en local**, sin GCP y sin Kubernetes. Esa es la ruta
de los pasos 1–5. GCP solo hace falta para el despliegue gestionado (pasos 6–8).

### Prerequisites

```bash
python --version          # >= 3.11 (obligatorio para todo lo demás)
docker --version          # Opcional: solo para el paso 5

# Solo para el despliegue en GCP (pasos 6-8)
gcloud --version          # Google Cloud SDK
terraform --version       # >= 1.6.0
kubectl version           # Kubernetes CLI
```

---

## 🖥️ Ejecución local (sin cloud)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/sentinel-platform.git
cd sentinel-platform

# Entorno virtual
python3.11 -m venv venv          # Windows: py -3.11 -m venv venv
source venv/bin/activate         # Windows: .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Todos los comandos que siguen se lanzan **desde la raíz del repositorio**, nunca desde `api/`.
> El paquete se importa como `api.main`, así que un `cd api` rompe los imports absolutos.

### 2. Entrenar los modelos

Ninguno de los dos tracks trae artefactos versionados: los `.joblib` no se suben al repositorio
(`models/flow/` entero está en `.gitignore`, y de `models/` solo se versionan los JSON de
métricas). En un clon limpio hay que entrenar.

El **track de eventos es obligatorio** — sin este paso, `/ready` devuelve 503 y `/v1/predict`
falla. Es un generador sintético, no necesita datos externos y tarda un par de minutos:

```bash
python -m ml.training.train --n-samples 60000 --attack-ratio 0.2 \
  --threshold-strategy fbeta --beta 2.0 --output-dir models --no-mlflow
```

Escribe `anomaly_detector.joblib`, `attack_classifier.joblib`, `metrics.json`,
`confusion_matrix.json` y `training_summary.json` en `models/`.

> El número de features del track de eventos (17, las de `FeatureExtractor`) **no es configurable
> a propósito**: es el mismo extractor que instancia `api/main.py` al arrancar, y esa identidad es
> lo único que impide servir un modelo cuyas columnas signifiquen otra cosa. Ver el comentario en
> [`ml/training/train.py`](./ml/training/train.py) donde vivía el flag `--use-window-features`.

El **track de flujos es opcional**: la API arranca sin él y `/ready` sigue devolviendo 200 con
`flow_detector: false`; son `/v1/predict/flow` y `/v1/analyze/flow/batch` los que responden 503
hasta que existan los artefactos. Entrenarlo **requiere los CSV de CICIDS2017 y ~8 GiB de RAM**:

```bash
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 \
  --pipeline xgb --split temporal --skip-tuning --output-dir models/flow
```

Los CSV (844 MB) **no se versionan**: se descargan de
[UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html), distribución `MachineLearningCSV`.
Detalles de opciones y resultados en [docs/ml/CICIDS2017-TRACK.md](./docs/ml/CICIDS2017-TRACK.md).

### 3. Arrancar la API

```bash
# Desde la raíz del repositorio
uvicorn api.main:app --reload --port 8000
```

Docs interactivas en <http://localhost:8000/docs>.

### 4. Comprobar que funciona

```bash
# 4a. ¿Vivo?
curl http://localhost:8000/health
# {"status":"healthy","timestamp":"..."}

# 4b. ¿Listo? Los tres checks del track de eventos deben dar true; los dos de flujos
#     solo si entrenaste ese track.
curl http://localhost:8000/ready
# {"status":"ready","checks":{"feature_extractor":true,"anomaly_detector":true,
#  "attack_classifier":true,"flow_detector":true,"flow_classifier":true}}
```

Un **503 en `/ready`** con `anomaly_detector: false` significa que falta el paso 2. Un
`flow_detector: false` **no** afecta al código de estado: el track de flujos es opcional y degrada
por su cuenta a 503 en sus propias rutas.

Los endpoints de predicción piden un JWT. Para desarrollo se firma con el secreto por defecto:

```bash
export TOKEN=$(python -c "from api.security.auth import create_access_token; \
  print(create_access_token('local-dev', scopes=['predict:read','predict:write']))")
```

```powershell
# PowerShell
$env:TOKEN = python -c "from api.security.auth import create_access_token; print(create_access_token('local-dev', scopes=['predict:read','predict:write']))"
```

Un evento de log y un flujo de red, uno por track:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"timestamp":"2024-01-15T14:32:18Z",
       "source":{"type":"auth","hostname":"server-01","ip":"10.0.1.45"},
       "actor":{"user":"admin","source_ip":"203.0.113.45"},
       "action":{"type":"authentication","subtype":"login_failed","result":"failure"}}'

curl -X POST http://localhost:8000/v1/predict/flow \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"features":{"Destination Port":80,"Flow Duration":1532000,"Total Fwd Packets":12}}'
```

Para una prueba con datos reales, cualquier fila de `api/csvs/*.csv` sirve como cuerpo de
`/v1/predict/flow`: se mandan las 78 columnas menos `Label` dentro de `features` y se compara la
respuesta con la etiqueta del CSV.

Y la suite de tests, que no necesita ni API levantada ni modelos entrenados:

```bash
pytest tests/unit -v
```

### 5. Docker (opcional)

```bash
docker build -f api/Dockerfile -t sentinel-api:local .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" sentinel-api:local
```

El montaje de `models/` es lo que da a la imagen los artefactos del paso 2; sin él la API arranca
igual pero con el track de eventos degradado.

---

## ☁️ Despliegue en GCP

### 6. GCP Setup

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

### 7. Deploy Infrastructure

```bash
cd infra/terraform/environments/dev
terraform init
terraform plan
terraform apply
```

### 8. Deploy to Kubernetes

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
>
> Los overlays de `k8s/overlays/` son la única pieza que **no** es cloud-agnóstica: dependen de ESO
> contra GCP Secret Manager. Para probarlos en un clúster local (kind/minikube) hay que sustituir
> el `ExternalSecret` por un `Secret` plano.

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
- [Troubleshooting de Secrets](./docs/SECRETS_TROUBLESHOOTING.md) - Problemas con ESO
- [Modulo Kubernetes Terraform](./infra/terraform/modules/kubernetes/README.md) - Configuracion de ESO
- [Technical Design Document](./sentinel_doc.md) - Diseno tecnico completo (infra, ML, API, k8s, CI/CD)
- [Documentacion de ML](./docs/ml/README.md) - Los dos tracks, resultados y su procedencia
- [Track de flujos (CICIDS2017)](./docs/ml/CICIDS2017-TRACK.md) - Cascada, contrato de features, particion temporal, entrenamiento

> Pendientes de escribir: `docs/adr/` (Architecture Decision Records) y `docs/runbooks/`.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.