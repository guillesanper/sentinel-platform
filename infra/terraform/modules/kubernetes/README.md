# Kubernetes Module

Módulo de Terraform para configurar recursos de Kubernetes en el cluster GKE de Sentinel Platform, incluyendo la instalación y configuración de External Secrets Operator (ESO).

## Descripción General

Este módulo automatiza la gestión segura de secrets en Kubernetes mediante:

1. **External Secrets Operator (ESO)**: Instalación via Helm
2. **ClusterSecretStore**: Conexión con GCP Secret Manager
3. **ExternalSecrets**: Sincronización automática de secrets a namespaces específicos
4. **Workload Identity**: Acceso seguro a GCP sin credenciales estáticas

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GCP Secret Manager                              │
│                                                                              │
│   ┌──────────────────────┐        ┌──────────────────────┐                  │
│   │ jwt-secret-{env}     │        │ api-key-{env}        │                  │
│   └──────────┬───────────┘        └──────────┬───────────┘                  │
│              │                               │                               │
└──────────────┼───────────────────────────────┼───────────────────────────────┘
               │                               │
               │     Workload Identity         │
               │    (sin credenciales)         │
               ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GKE Cluster                                     │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Namespace: external-secrets                       │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │           External Secrets Operator (ESO)                    │   │   │
│   │   │         Service Account con Workload Identity                │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                     ┌──────────────┴──────────────┐                         │
│                     ▼                              ▼                         │
│   ┌─────────────────────────────┐  ┌─────────────────────────────┐         │
│   │   Namespace: platform       │  │   Namespace: ml             │         │
│   │                             │  │                             │         │
│   │   ExternalSecret            │  │   ExternalSecret            │         │
│   │       ↓                     │  │       ↓                     │         │
│   │   sentinel-secrets          │  │   sentinel-secrets          │         │
│   │   - jwt-secret              │  │   - jwt-secret              │         │
│   │   - api-key                 │  │   - api-key                 │         │
│   │       ↓                     │  │       ↓                     │         │
│   │   [Pods consumen secrets]   │  │   [Pods consumen secrets]   │         │
│   └─────────────────────────────┘  └─────────────────────────────┘         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Uso

### Ejemplo Básico

```hcl
module "kubernetes" {
  source = "../../modules/kubernetes"

  project_id  = "mi-proyecto-gcp"
  region      = "europe-west1"
  environment = "dev"

  # Conexión al cluster GKE
  cluster_name           = module.gke.cluster_name
  cluster_endpoint       = module.gke.cluster_endpoint
  cluster_ca_certificate = module.gke.cluster_ca_certificate

  # Secrets a sincronizar
  secrets_to_sync   = ["jwt-secret", "api-key"]
  target_namespaces = ["platform", "ml"]

  depends_on = [module.gke, module.security]
}
```

### Ejemplo con Configuración Avanzada

```hcl
module "kubernetes" {
  source = "../../modules/kubernetes"

  project_id  = var.project_id
  region      = var.region
  environment = "prod"

  cluster_name           = module.gke.cluster_name
  cluster_endpoint       = module.gke.cluster_endpoint
  cluster_ca_certificate = module.gke.cluster_ca_certificate

  # Configuración de ESO
  install_eso   = true
  eso_namespace = "external-secrets"
  eso_version   = "0.9.11"

  # Secrets personalizados
  secrets_to_sync = [
    "jwt-secret",
    "api-key",
    "database-password",
    "redis-password"
  ]

  # Namespaces destino
  target_namespaces = ["platform", "ml", "workers"]

  # Service account existente (opcional)
  eso_service_account_email = "eso-sa@mi-proyecto.iam.gserviceaccount.com"

  labels = {
    project     = "sentinel"
    environment = "prod"
    managed_by  = "terraform"
  }

  depends_on = [module.gke, module.security, module.iam]
}
```

## Variables de Entrada

| Variable | Tipo | Requerido | Default | Descripción |
|----------|------|-----------|---------|-------------|
| `project_id` | `string` | Sí | - | ID del proyecto GCP |
| `environment` | `string` | Sí | - | Entorno (dev, staging, prod) |
| `region` | `string` | Sí | - | Región GCP |
| `cluster_name` | `string` | Sí | - | Nombre del cluster GKE |
| `cluster_endpoint` | `string` | Sí | - | Endpoint del cluster GKE |
| `cluster_ca_certificate` | `string` | Sí | - | Certificado CA del cluster (base64) |
| `install_eso` | `bool` | No | `true` | Instalar External Secrets Operator |
| `eso_namespace` | `string` | No | `external-secrets` | Namespace para ESO |
| `eso_version` | `string` | No | `0.9.11` | Versión del chart Helm de ESO |
| `secrets_to_sync` | `list(string)` | No | `["jwt-secret", "api-key"]` | Secrets a sincronizar desde GCP |
| `target_namespaces` | `list(string)` | No | `["platform", "ml"]` | Namespaces donde crear los secrets |
| `eso_service_account_email` | `string` | No | `""` | Email del SA existente (si está vacío, se crea uno) |
| `labels` | `map(string)` | No | `{}` | Labels para los recursos |

## Outputs

| Output | Descripción |
|--------|-------------|
| `eso_namespace` | Namespace donde está instalado ESO |
| `cluster_secret_store_name` | Nombre del ClusterSecretStore |
| `synced_secrets` | Lista de secrets sincronizados |
| `target_namespaces` | Namespaces donde se sincronizan los secrets |
| `eso_service_account` | Service account de Kubernetes usado por ESO |
| `verify_commands` | Comandos útiles para verificar la instalación |

## Recursos Creados

### En GCP

| Recurso | Descripción |
|---------|-------------|
| `google_service_account.eso` | Service Account para ESO |
| `google_project_iam_member.eso_secret_accessor` | Permiso `secretmanager.secretAccessor` |
| `google_service_account_iam_member.eso_workload_identity` | Binding de Workload Identity |

### En Kubernetes

| Recurso | Descripción |
|---------|-------------|
| `kubernetes_namespace.eso` | Namespace `external-secrets` |
| `helm_release.external_secrets` | Chart Helm de ESO |
| `kubectl_manifest.cluster_secret_store` | ClusterSecretStore para GCP |
| `kubectl_manifest.external_secrets` | ExternalSecrets por namespace |

## Convención de Nombres de Secrets

Los secrets en GCP Secret Manager deben seguir el formato:

```
{nombre-secret}-{environment}
```

Ejemplos:
- `jwt-secret-dev`
- `jwt-secret-prod`
- `api-key-dev`
- `api-key-prod`

El módulo añade automáticamente el sufijo `-{environment}` al buscar los secrets en GCP.

## Prerrequisitos

1. **API de Secret Manager habilitada**:
   ```bash
   gcloud services enable secretmanager.googleapis.com --project=TU_PROYECTO
   ```

2. **Secrets creados en GCP Secret Manager**:
   ```bash
   # Crear los secrets (sin valor inicial)
   gcloud secrets create jwt-secret-dev --project=TU_PROYECTO
   gcloud secrets create api-key-dev --project=TU_PROYECTO

   # Añadir valores
   echo -n "mi-jwt-secret-super-seguro" | \
     gcloud secrets versions add jwt-secret-dev --data-file=- --project=TU_PROYECTO

   echo -n "mi-api-key-super-segura" | \
     gcloud secrets versions add api-key-dev --data-file=- --project=TU_PROYECTO
   ```

3. **Workload Identity habilitado** en el cluster GKE (configurado por el módulo GKE)

## Verificación Post-Deployment

Después de ejecutar `terraform apply`, verifica la instalación:

```bash
# 1. Verificar pods de ESO
kubectl get pods -n external-secrets
# Esperado: external-secrets-xxxxx (Running)

# 2. Verificar ClusterSecretStore
kubectl get clustersecretstore
# Esperado: gcp-secret-manager (Ready)

# 3. Verificar ExternalSecrets
kubectl get externalsecrets -A
# Esperado: sentinel-secrets en platform y ml (SecretSynced)

# 4. Verificar secrets creados
kubectl get secrets sentinel-secrets -n platform -o yaml
kubectl get secrets sentinel-secrets -n ml -o yaml

# 5. Ver logs de ESO si hay problemas
kubectl logs -n external-secrets -l app.kubernetes.io/name=external-secrets
```

## Integración con Deployments

Los pods pueden consumir los secrets sincronizados de varias formas:

### Como Variables de Entorno

```yaml
spec:
  containers:
    - name: mi-app
      env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: sentinel-secrets
              key: jwt-secret
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: sentinel-secrets
              key: api-key
```

### Como Volumen Montado

```yaml
spec:
  containers:
    - name: mi-app
      volumeMounts:
        - name: secrets
          mountPath: "/etc/secrets"
          readOnly: true
  volumes:
    - name: secrets
      secret:
        secretName: sentinel-secrets
```

## Rotación de Secrets

ESO sincroniza automáticamente los cambios de GCP Secret Manager cada **1 hora** (configurable via `refreshInterval`).

Para forzar una sincronización inmediata:

```bash
# Eliminar el secret local (ESO lo recreará)
kubectl delete secret sentinel-secrets -n platform

# O anotar el ExternalSecret para trigger refresh
kubectl annotate externalsecret sentinel-secrets -n platform \
  force-sync=$(date +%s) --overwrite
```

## Seguridad

- **Sin credenciales estáticas**: Usa Workload Identity
- **Principio de mínimo privilegio**: Solo acceso `secretAccessor`
- **Secrets nunca en Git**: Los valores solo existen en GCP Secret Manager
- **Auditoría**: Todos los accesos quedan registrados en Cloud Audit Logs

## Notas Importantes

1. **Dependencias**: Este módulo debe ejecutarse DESPUÉS de los módulos `gke` y `security`

2. **Namespaces**: Los namespaces target (`platform`, `ml`) deben existir antes de aplicar los ExternalSecrets. Kustomize los crea automáticamente.

3. **Tiempo de sincronización**: ESO puede tardar hasta 30 segundos en sincronizar secrets tras `terraform apply`

4. **CRDs**: El chart de ESO instala los CRDs automáticamente (`installCRDs: true`)

## Troubleshooting

Ver la [guía de troubleshooting](../../../../docs/SECRETS_TROUBLESHOOTING.md) para problemas comunes.
