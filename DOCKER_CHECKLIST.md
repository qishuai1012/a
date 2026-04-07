# Docker 部署检查清单

## ✅ 已修复的问题

1. **Dockerfile CMD 指令**
   - 已修复：添加 bash 到系统依赖
   - 已修复：修改 CMD 为动态服务类型检测，支持微服务模式

2. **环境变量安全性**
   - 已修复：`OPENAI_API_KEY` 添加了强制检查 `:?OPENAI_API_KEY must be set`
   - 已修复：`JWT_SECRET_KEY` 保持强制检查（已有）
   - 已修复：`MINIO_ROOT_PASSWORD` 添加了默认值，便于开发

3. **etcd 配置**
   - 已修复：`advertise-client-urls` 从 `127.0.0.1` 改为 `etcd`（容器名），允许其他容器访问

4. **未使用的卷**
   - 已修复：删除了未使用的 `vector_store_data` 卷

5. **只读挂载**
   - 已修复：Prometheus 配置文件改为只读挂载 `:ro`

6. **内存配置**
   - 已修复：移除了 Elasticsearch 重复的内存配置

## ⚠️ 需要注意的事项

### 1. 环境文件

- 已创建 `.env.example`，请复制为 `.env` 并配置
- **重要**：生产环境必须修改所有默认密码和密钥

### 2. 服务端口映射

当前配置暴露了所有端口到主机，生产环境建议：
- 只暴露必要的端口（如 8000、9001）
- 使用内部网络通信，避免外部直接访问

### 3. 缺失的微服务

Docker Compose 中定义了以下服务但未在 `microservice_main.py` 中实现：
- `llm_service` - LLM 服务
- `document_processor_service` - 文档处理服务
- `storage_service` - 存储服务

**建议**：
- 如果这些服务未实现，可以从 `docker-compose.yml` 中移除
- 或者在 `microservice_main.py` 中添加对应的实现

### 4. Milvus 配置

Milvus 依赖 etcd 和 minio，已正确配置依赖关系：
```yaml
depends_on:
  - etcd
  - minio
```

但需要注意：
- etcd 和 minio 启动可能需要时间
- 可能需要添加 healthcheck 确保依赖服务就绪

### 5. 数据持久化

已配置的持久化卷：
- `document_storage` - 文档存储
- `cache_data` - 缓存数据
- `auth_data` - 认证数据
- `redis_data` - Redis 数据
- `es_data` - Elasticsearch 数据
- `milvus_data` - Milvus 数据
- `etcd_data` - etcd 数据
- `minio_data` - MinIO 数据

**建议**：定期备份这些卷

### 6. 网络配置

所有服务使用同一网络 `qa_network`，这是正确的。

### 7. 资源限制

未配置资源限制，生产环境建议添加：

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '0.5'
      memory: 1G
```

## 📋 部署步骤

1. **准备环境**
   ```bash
   cp .env.example .env
   # 编辑 .env，修改 JWT_SECRET_KEY 和 OPENAI_API_KEY
   ```

2. **构建镜像**
   ```bash
   docker-compose build
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

4. **检查状态**
   ```bash
   docker-compose ps
   ```

5. **查看日志**
   ```bash
   docker-compose logs -f
   ```

6. **测试健康检查**
   ```bash
   curl http://localhost:8000/health
   ```

## 🔍 潜在问题

### 1. 服务启动顺序

Docker Compose 的 `depends_on` 只保证容器启动顺序，不保证服务就绪。建议为关键服务添加 healthcheck：

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:${PORT}/health"]
  interval: 30s
  timeout: 10s
  retries: 5
```

### 2. 开发与生产环境分离

当前 `docker-compose.yml` 同时包含开发和生产服务（如 Prometheus、Kibana、Elasticsearch）。建议创建：
- `docker-compose.dev.yml` - 开发环境
- `docker-compose.prod.yml` - 生产环境

### 3. 密钥管理

当前使用环境变量管理密钥，生产环境建议使用：
- Docker Secrets
- Kubernetes Secrets
- HashiCorp Vault

### 4. 监控告警

已配置 Prometheus，但缺少：
- 告警规则
- Grafana 仪表板
- 告警通知配置（邮件、Slack等）

## ✅ 最终检查

- [x] Dockerfile 支持微服务模式
- [x] 所有必需的环境变量都有检查
- [x] 服务依赖关系正确配置
- [x] 数据持久化卷正确配置
- [x] 网络配置正确
- [ ] 建议添加 healthcheck
- [ ] 建议分离开发/生产配置
- [ ] 建议配置资源限制
- [ ] 建议配置监控告警

## 📝 总结

**Docker 部署文件基本正确**，已修复主要问题。但仍有一些优化建议：

1. **短期**：添加 healthcheck 确保服务就绪
2. **中期**：实现缺失的微服务或从 compose 文件中移除
3. **长期**：分离开发/生产配置，完善监控告警

可以开始部署测试了！
