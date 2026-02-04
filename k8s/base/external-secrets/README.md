# External Secrets

External Secrets are managed by Terraform in `infra/terraform/modules/kubernetes/`.

## What Terraform Creates

1. **External Secrets Operator (ESO)** - Installed via Helm in `external-secrets` namespace
2. **ClusterSecretStore** - Named `gcp-secret-manager`, connects to GCP Secret Manager
3. **ExternalSecrets** - In `platform` and `ml` namespaces, syncing secrets from GCP

## Secrets Synced

| GCP Secret Manager | K8s Secret Key | Namespaces |
|--------------------|----------------|------------|
| `jwt-secret-{env}` | `jwt-secret` | platform, ml |
| `api-key-{env}` | `api-key` | platform, ml |

## Verification Commands

```bash
# Check ESO is running
kubectl get pods -n external-secrets

# Check ClusterSecretStore
kubectl get clustersecretstore

# Check ExternalSecrets status
kubectl get externalsecrets -A

# Check synced secrets
kubectl get secrets sentinel-secrets -n platform -o yaml
```

## Populating Secrets in GCP

After `terraform apply`, populate the secrets:

```bash
# Set your project ID
export PROJECT_ID=your-project-id
export ENVIRONMENT=dev

# Add JWT secret
echo -n "your-jwt-secret-value" | gcloud secrets versions add jwt-secret-${ENVIRONMENT} \
  --data-file=- --project=${PROJECT_ID}

# Add API key
echo -n "your-api-key-value" | gcloud secrets versions add api-key-${ENVIRONMENT} \
  --data-file=- --project=${PROJECT_ID}
```

ESO will automatically sync the secrets to Kubernetes within the refresh interval (1 hour by default).
