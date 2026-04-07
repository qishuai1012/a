# Docker Deployment for Enterprise QA System

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Git
- 8GB+ RAM (16GB+ recommended)
- 20GB+ disk space

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd enterprise-qa-system
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and update the following values:

```bash
# Change these for production
JWT_SECRET_KEY=your-secure-random-key
OPENAI_API_KEY=sk-your-openai-api-key
MINIO_ROOT_PASSWORD=change-this-password
```

### 3. Build and Start Services

Start all services in detached mode:

```bash
docker-compose up -d
```

### 4. Verify Deployment

Check that all services are running:

```bash
docker-compose ps
```

Expected services:
- `api_gateway` (port 8000)
- `auth_service` (port 8001)
- `qa_service` (port 8002)
- `vector_store_service` (port 8004)
- `bm25_cache_service` (port 8005)
- `document_processor_service` (port 8006)
- `storage_service` (port 8007)
- `llm_service` (port 8008)
- `redis` (port 6379)
- `milvus-standalone` (port 19530)
- `etcd` (port 2379)
- `minio` (port 9000, console on 9001)
- `prometheus` (port 9090)
- `elasticsearch` (port 9200)
- `kibana` (port 5601)

### 5. Access Services

- **API Gateway**: http://localhost:8000
- **MinIO Console**: http://localhost:9001 (login: admin/minioadmin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Milvus**: Connect on port 19530

## Service Details

### Core Services

#### API Gateway (port 8000)
Main entry point for all API requests.

#### Authentication Service (port 8001)
JWT-based authentication and authorization.

#### QA Service (port 8002)
Main QA system with RAG pipeline.

#### Vector Store Service (port 8004)
Milvus-based vector storage.

#### BM25 Cache Service (port 8005)
Enhanced query caching layer.

#### Document Processor (port 8006)
Document processing and chunking.

#### Storage Service (port 8007)
Document storage management.

#### LLM Service (port 8008)
LLM API integration (OpenAI).

### Infrastructure Services

#### Redis (port 6379)
Distributed caching and rate limiting.

#### Milvus (port 19530)
Vector database for embeddings.

#### MinIO (port 9000/9001)
S3-compatible object storage.

#### Etcd (port 2379)
Distributed key-value store for Milvus.

#### Prometheus (port 9090)
Metrics monitoring.

#### Elasticsearch (port 9200)
Logging and analytics.

#### Kibana (port 5601)
Log visualization.

## Common Operations

### View Logs

All services:
```bash
docker-compose logs -f
```

Specific service:
```bash
docker-compose logs -f api_gateway
```

### Stop Services

```bash
docker-compose down
```

### Stop Services and Remove Volumes

```bash
docker-compose down -v
```

### Rebuild Services

```bash
docker-compose build --no-cache
docker-compose up -d
```

### Restart a Service

```bash
docker-compose restart qa_service
```

### Scale Services

```bash
docker-compose up -d --scale qa_service=3
```

## Ingesting Documents

### Using the CLI

```bash
docker-compose exec qa_service python main.py ingest /path/to/documents
```

### Via API

```bash
curl -X POST http://localhost:8002/ingest \
  -F "file=@document.pdf" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Querying

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "What is RAG?"}'
```

## Health Checks

```bash
curl http://localhost:8000/health
```

## Production Deployment

### Security Considerations

1. **Change all default passwords** in `.env`
2. **Use HTTPS** for all external traffic
3. **Enable JWT secret rotation**
4. **Configure network policies**
5. **Use Docker secrets** for sensitive data
6. **Enable rate limiting**
7. **Set up monitoring alerts**

### Resource Requirements

Minimum for production:
- 16GB RAM
- 4 CPU cores
- 100GB SSD storage
- 1Gbps network

### Scaling

For high availability:

1. **API Gateway**: Scale horizontally
2. **QA Service**: Scale based on query load
3. **Vector Store**: Use Milvus cluster mode
4. **Database**: Use Redis cluster
5. **Storage**: Use distributed MinIO

Example scaling:
```bash
docker-compose up -d --scale api_gateway=2 --scale qa_service=4
```

### Backup Strategy

**Milvus data**:
```bash
docker-compose exec milvus-standalone bash -c "milvus backup create --name backup-$(date +%Y%m%d)"
```

**MinIO data**:
```bash
docker-compose exec minio mc mirror minio/data /backup/minio
```

**Redis data**:
```bash
docker-compose exec redis BGSAVE
```

**Elasticsearch data**:
```bash
curl -X PUT "localhost:9200/_snapshot/backup" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/usr/share/elasticsearch/backups"
  }
}'
```

## Troubleshooting

### Services not starting

Check logs:
```bash
docker-compose logs <service_name>
```

Check dependencies:
```bash
docker-compose ps
```

### Milvus issues

Check etcd and minio:
```bash
docker-compose logs etcd
docker-compose logs minio
```

### Port conflicts

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "9000:9000"  # external:internal
```

### High memory usage

Limit memory per service:
```yaml
deploy:
  resources:
    limits:
      memory: 2g
```

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Common queries:
- Service health: `up`
- Request rate: `rate(http_requests_total[5m])`
- Error rate: `rate(http_requests_total{status=~"5.."}[5m])`

### Kibana Logs

Access Kibana at http://localhost:5601

Create index pattern: `logstash-*`

## Development vs Production

### Development

```bash
docker-compose up -d
```

### Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  api_gateway:
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2g
        reservations:
          memory: 1g
```

Deploy:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Cleanup

### Stop all services

```bash
docker-compose down
```

### Remove all data

```bash
docker-compose down -v
```

### Remove all containers and images

```bash
docker system prune -a
```

## Next Steps

1. Configure authentication and get JWT token
2. Ingest your first document
3. Test querying
4. Set up monitoring dashboards
5. Configure backup strategy
6. Plan for scaling
