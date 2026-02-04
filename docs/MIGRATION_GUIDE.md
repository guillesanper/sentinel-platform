# Guía de Migración: deploy.sh a Terraform + ESO

Esta guía explica cómo migrar del antiguo sistema basado en `deploy.sh` + `.env` al nuevo sistema basado en Terraform + External Secrets Operator.

## Resumen de Cambios

| Aspecto | Antes (deploy.sh) | Ahora (Terraform + ESO) |
|---------|-------------------|-------------------------|
| Almacenamiento de secrets | Archivo `.env` local | GCP Secret Manager |
| Sustitución de variables | `envsubst` en runtime | ESO sincroniza automáticamente |
| Gestión de secrets K8s | `secretGenerator` en Kustomize | ExternalSecrets via ESO |
| Script de deployment | `scripts/deploy.sh` | `scripts/deploy-k8s.sh` |
| Configuración de entorno | Variables en `.env` | Placeholders en overlays |
| Seguridad | Secrets en texto plano | Cifrado en GCP + Workload Identity |

## Arquitectura Comparada

### Antes: deploy.sh + .env

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   .env file     │────►│   deploy.sh     │────►│  K8s Manifests  │
│                 │     │   + envsubst    │     │  (con valores)  │
│ JWT_SECRET=xxx  │     │                 │     │                 │
│ API_KEY=yyy     │     │                 │     │                 │
│ PROJECT_ID=zzz  │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
   ⚠️ Riesgo de seguridad:
   - Secrets en texto plano
   - Posible commit accidental
   - Sin auditoría de acceso
```

### Ahora: Terraform + External Secrets Operator

```
┌─────────────────────────────────────────────────────────────────┐
│                    GCP Secret Manager                            │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ jwt-secret-dev  │  │  api-key-dev    │                       │
│  └────────┬────────┘  └────────┬────────┘                       │
│           │                    │                                 │
└───────────┼────────────────────┼─────────────────────────────────┘
            │  Workload Identity │
            │    (sin keys)      │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GKE Cluster                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            External Secrets Operator (ESO)               │    │
│  │         Instalado via Terraform + Helm                   │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              K8s Secret: sentinel-secrets                │    │
│  │            (creado automáticamente por ESO)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ✅ Beneficios:                                                  │
│  - Secrets cifrados en reposo                                   │
│  - Auditoría completa                                           │
│  - Rotación automática                                          │
│  - Sin credenciales en código                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Pasos de Migración

### Paso 1: Backup de Secrets Actuales

Antes de migrar, guarda los valores actuales de tu `.env`:

```bash
# Copia el archivo .env actual (NO lo subas a git)
cp .env .env.backup

# Verifica los valores
cat .env.backup
# Deberías ver algo como:
# JWT_SECRET=mi-jwt-secret-actual
# API_KEY=mi-api-key-actual
# PROJECT_ID=mi-proyecto-gcp
```

### Paso 2: Eliminar Archivo .env del Proyecto

```bash
# Verifica que .env está en .gitignore
grep "^\.env$" .gitignore || echo ".env" >> .gitignore

# Elimina el archivo .env (ya tienes backup)
rm .env
```

### Paso 3: Aplicar Terraform

Terraform creará la infraestructura necesaria incluyendo ESO:

```bash
cd infra/terraform/environments/dev

# Inicializar (si no lo has hecho)
terraform init

# Revisar cambios
terraform plan -var="project_id=TU_PROYECTO"

# Aplicar
terraform apply -var="project_id=TU_PROYECTO"
```

Esto creará:
- Secrets vacíos en GCP Secret Manager (`jwt-secret-dev`, `api-key-dev`)
- External Secrets Operator en el cluster
- ClusterSecretStore conectado a GCP
- ExternalSecrets en namespaces `platform` y `ml`

### Paso 4: Migrar Valores de Secrets a GCP

Usa los valores de tu backup para poblar GCP Secret Manager:

```bash
export PROJECT_ID=tu-proyecto-id
export ENVIRONMENT=dev

# Lee los valores de tu backup
source .env.backup

# Migra JWT_SECRET a GCP
echo -n "$JWT_SECRET" | gcloud secrets versions add jwt-secret-${ENVIRONMENT} \
  --data-file=- --project=${PROJECT_ID}

# Migra API_KEY a GCP
echo -n "$API_KEY" | gcloud secrets versions add api-key-${ENVIRONMENT} \
  --data-file=- --project=${PROJECT_ID}

# Verifica que se crearon correctamente
gcloud secrets versions list jwt-secret-${ENVIRONMENT} --project=${PROJECT_ID}
gcloud secrets versions list api-key-${ENVIRONMENT} --project=${PROJECT_ID}
```

### Paso 5: Actualizar Overlays de Kustomize

Reemplaza `PROJECT_ID` en los overlays:

```bash
# Para dev
sed -i "s/PROJECT_ID/${PROJECT_ID}/g" k8s/overlays/dev/kustomization.yaml

# Para prod (si aplica)
sed -i "s/PROJECT_ID/${PROJECT_ID}/g" k8s/overlays/prod/kustomization.yaml
```

### Paso 6: Verificar Sincronización de ESO

```bash
# Espera a que ESO sincronice (puede tardar hasta 30 segundos)
sleep 30

# Verifica que los ExternalSecrets están sincronizados
kubectl get externalsecrets -A

# Esperado:
# NAMESPACE   NAME              STORE                REFRESH   STATUS         READY
# platform    sentinel-secrets  gcp-secret-manager   1h        SecretSynced   True
# ml          sentinel-secrets  gcp-secret-manager   1h        SecretSynced   True

# Verifica que los secrets K8s existen
kubectl get secret sentinel-secrets -n platform
kubectl get secret sentinel-secrets -n ml
```

### Paso 7: Redesplegar la Aplicación

Usa el nuevo script de deployment:

```bash
# Nuevo script (simplificado)
./scripts/deploy-k8s.sh dev

# O directamente con kubectl
kubectl apply -k k8s/overlays/dev
```

### Paso 8: Verificar que Todo Funciona

```bash
# Verifica pods
kubectl get pods -n platform

# Verifica que los secrets están disponibles
kubectl exec -it deployment/sentinel-api -n platform -- env | grep -E "JWT|API"

# Test de salud
kubectl port-forward svc/sentinel-api 8000:80 -n platform &
curl http://localhost:8000/health
```

### Paso 9: Limpieza

Una vez verificado que todo funciona:

```bash
# Elimina el backup de .env de forma segura
shred -u .env.backup 2>/dev/null || rm -f .env.backup

# Elimina el antiguo deploy.sh (ya no es necesario)
# NOTA: Este archivo ya fue eliminado en el commit actual
```

## Mapeo de Variables

| Variable .env (antes) | Secret GCP (ahora) | Key en K8s Secret |
|-----------------------|--------------------|--------------------|
| `JWT_SECRET` | `jwt-secret-{env}` | `jwt-secret` |
| `API_KEY` | `api-key-{env}` | `api-key` |
| `PROJECT_ID` | N/A (en overlay) | N/A |
| `ENVIRONMENT` | N/A (en overlay) | N/A |

## Comparación de Scripts

### Antes: deploy.sh

```bash
#!/bin/bash
# Cargaba .env y usaba envsubst
set -a
source .env
set +a

# Procesaba templates
envsubst < k8s/base/deployment.yaml | kubectl apply -f -
envsubst < k8s/base/configmap.yaml | kubectl apply -f -

# Aplicaba secrets inline
kubectl create secret generic sentinel-secrets \
  --from-literal=jwt-secret=$JWT_SECRET \
  --from-literal=api-key=$API_KEY \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Ahora: deploy-k8s.sh

```bash
#!/bin/bash
# Simplificado - solo aplica Kustomize
set -euo pipefail

ENVIRONMENT="${1:-dev}"
kubectl apply -k "k8s/overlays/$ENVIRONMENT"

# Secrets ya están sincronizados por ESO
kubectl get externalsecrets -n platform
```

## Diferencias en Kustomization

### Antes

```yaml
# k8s/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Secrets generados por Kustomize (inseguro)
secretGenerator:
  - name: sentinel-secrets
    literals:
      - jwt-secret=${JWT_SECRET}   # Requería envsubst
      - api-key=${API_KEY}         # Requería envsubst
```

### Ahora

```yaml
# k8s/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Sin secretGenerator - ESO crea los secrets automáticamente
# Solo configuramos las referencias en los deployments
patches:
  - patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: sentinel-api
      spec:
        template:
          spec:
            containers:
              - name: sentinel-api
                env:
                  - name: JWT_SECRET
                    valueFrom:
                      secretKeyRef:
                        name: sentinel-secrets  # Creado por ESO
                        key: jwt-secret
```

## FAQ de Migración

### ¿Qué pasa con mis secrets actuales en el cluster?

Los secrets antiguos creados por `secretGenerator` seguirán existiendo hasta que los elimines manualmente o ESO los sobrescriba (si tienen el mismo nombre).

```bash
# Ver secrets actuales
kubectl get secrets -n platform

# Eliminar secret antiguo si existe
kubectl delete secret sentinel-secrets-HASH -n platform
```

### ¿Puedo seguir usando .env para desarrollo local?

Sí, para desarrollo local puedes seguir usando `.env` con `python-dotenv` o similar. El cambio solo afecta a deployments en Kubernetes.

### ¿Qué pasa si necesito añadir un nuevo secret?

1. Crea el secret en GCP Secret Manager:
   ```bash
   gcloud secrets create nuevo-secret-dev --project=TU_PROYECTO
   echo -n "valor" | gcloud secrets versions add nuevo-secret-dev --data-file=-
   ```

2. Añádelo a la variable `secrets_to_sync` en Terraform:
   ```hcl
   module "kubernetes" {
     secrets_to_sync = ["jwt-secret", "api-key", "nuevo-secret"]
   }
   ```

3. Aplica Terraform:
   ```bash
   terraform apply
   ```

### ¿Cómo roto un secret?

1. Actualiza el valor en GCP Secret Manager:
   ```bash
   echo -n "nuevo-valor" | gcloud secrets versions add jwt-secret-dev --data-file=-
   ```

2. ESO sincronizará automáticamente en la próxima iteración (1 hora) o fuerza la sincronización:
   ```bash
   kubectl annotate externalsecret sentinel-secrets -n platform \
     force-sync=$(date +%s) --overwrite
   ```

3. Reinicia los pods para que tomen el nuevo valor:
   ```bash
   kubectl rollout restart deployment/sentinel-api -n platform
   ```

## Rollback a deploy.sh

Si necesitas volver al sistema anterior temporalmente:

```bash
# 1. Restaura tu archivo .env
cp .env.backup .env

# 2. Restaura deploy.sh desde git
git checkout HEAD~1 -- scripts/deploy.sh

# 3. Ejecuta el script antiguo
./scripts/deploy.sh
```

**Nota**: Esto no es recomendado para producción. El nuevo sistema es más seguro.

## Checklist de Migración

- [ ] Backup de `.env` realizado
- [ ] `.env` añadido a `.gitignore`
- [ ] `terraform apply` ejecutado exitosamente
- [ ] Secrets migrados a GCP Secret Manager
- [ ] ESO instalado y funcionando
- [ ] ExternalSecrets sincronizados (STATUS: SecretSynced)
- [ ] `PROJECT_ID` reemplazado en overlays
- [ ] Aplicación redesplegada con nuevo script
- [ ] Verificado que pods tienen acceso a secrets
- [ ] Backup de `.env` eliminado de forma segura
- [ ] Antiguo `deploy.sh` eliminado
