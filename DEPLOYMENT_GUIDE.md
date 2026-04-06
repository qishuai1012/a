# 企业级RAG系统 - 部署指南

## 项目概述

这是一个企业级的私有知识库问答系统，采用RAG（检索增强生成）架构，具备BM25缓存、混合检索、完整安全控制和微服务架构。

## 架构特性

### 1. 微服务架构
- **API网关**：统一入口、路由和负载均衡
- **认证服务**：JWT认证和RBAC权限控制
- **QA服务**：核心问答逻辑处理
- **向量存储服务**：**Milvus企业级向量数据库**（默认），支持ChromaDB（备选）
- **BM25缓存服务**：多级缓存机制
- **文档处理服务**：文档解析和索引

### 2. 安全特性
- **JWT认证**：安全的令牌管理
- **RBAC权限控制**：基于角色的访问控制
- **速率限制**：令牌桶算法防止API滥用
- **熔断机制**：防止级联故障
- **数据加密**：敏感信息保护

### 3. 性能优化
- **多级缓存**：内存→Redis→持久化
- **异步处理**：高并发支持
- **向量化加速**：使用Milvus实现快速语义检索
- **混合检索**：向量+关键词结合
- **RRF融合算法**：优化检索结果排序

### 4. 可扩展性（企业级）
- **Milvus向量数据库**：支持大规模数据集、分布式部署、高级查询功能
- **水平扩展**：支持多节点部署和负载分发
- **高可用性**：内置复制和故障转移机制

### 5. 可观测性
- **结构化日志**：统一日志格式
- **性能指标**：Prometheus监控
- **健康检查**：服务状态监控
- **错误追踪**：全面错误处理

## 部署方式

### 容器化部署（推荐）

#### 1. 使用Docker Compose部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 2. 单独部署各服务

```bash
# 启动API网关
docker-compose up -d api-gateway

# 启动认证服务
docker-compose up -d auth-service

# 启动QA服务
docker-compose up -d qa-service

# 启动向量存储服务
docker-compose up -d vector-store

# 启动BM25缓存服务
docker-compose up -d bm25-cache
```

### 传统部署

#### 1. 环境准备

```bash
# Python版本要求
python --version  # 需要Python 3.10+

# 安装依赖
pip install -r requirements.txt
```

#### 2. 服务部署

```bash
# 安装项目
pip install -e .

# 运行API服务器
python main.py server --host 0.0.0.0 --port 8000

# 或者在微服务模式下运行
# 启动API网关
python main.py microservice gateway --port 8000

# 启动QA服务
python main.py microservice qa --port 8002

# 启动认证服务
python main.py microservice auth --port 8001

# 启动向量存储服务（需要先启动Milvus）
python main.py microservice vector_store --port 8004

# 启动BM25缓存服务
python main.py microservice bm25_cache --port 8005
```

## 环境配置

### 1. Milvus配置

需要先启动Milvus服务：

```bash
# 如果使用独立部署，需要先启动Milvus
# 参考官方文档：https://milvus.io/docs/install_standalone-docker.md
```

### 2. Redis配置

确保Redis服务可用，用于分布式缓存和会话管理。

### 3. 环境变量

创建`.env`文件配置系统参数：

```bash
# JWT配置
JWT_SECRET_KEY=your-super-secret-key-change-this
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# 向量数据库配置
VECTOR_STORE_PROVIDER=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis配置
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# LLM配置
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://api.example.com/v1

# 文档处理配置
DEFAULT_BUSINESS_CATEGORY=general
DEFAULT_PERMISSION_LEVEL=public
```

## API端点

### 认证
- `POST /login` - 用户登录

### 问答功能
- `POST /query` - 查询接口（需认证）
- `POST /query/stream` - 流式查询接口（需认证）
- `POST /ingest/file` - 文件注入（需认证+写权限）
- `POST /ingest/directory` - 目录注入（需认证+写权限）

### 系统管理
- `GET /health` - 健康检查
- `GET /metrics` - 性能指标
- `GET /profile` - 用户资料
- `GET /session/{session_id}` - 会话信息
- `DELETE /session/{session_id}` - 删除会话

### 管理员接口
- `GET /admin/stats` - 系统统计（管理员权限）

## 监控和日志

### 1. 日志配置

系统使用structlog记录结构化日志，支持多种输出格式。

### 2. 性能监控

- Prometheus指标收集
- 请求量、响应时间、错误率监控
- 向量检索性能监控
- 缓存命中率统计

### 3. 健康检查

定期检查服务状态和依赖项（数据库、缓存等）。

## 安全配置

### 1. 认证流程

1. 用户通过`/login`端点获取JWT令牌
2. 后续请求在Header中携带`Authorization: Bearer <token>`
3. 系统验证令牌有效性和权限

### 2. 权限控制

- RBAC模型支持角色和权限管理
- 不同端点有不同的权限要求
- 文档级别的权限控制

### 3. 速率限制

- 令牌桶算法防止滥用
- 不同用户角色有不同的限流策略
- IP级别的限流保护

## 扩展和定制

### 1. 添加新的文档类型

在`src/document_loader/loader.py`中添加新的加载器类。

### 2. 自定义检索算法

在`src/rag_system/retriever.py`中扩展检索逻辑。

### 3. 集成新的向量数据库

实现`VectorStore`接口即可接入新的向量数据库。

## 维护和运维

### 1. 定期维护

- 定期备份向量数据库
- 监控API性能指标
- 检查系统健康状态
- 更新安全证书和密钥

### 2. 数据备份

```bash
# 备份向量数据库
# 具体步骤取决于使用的向量数据库

# 备份缓存数据
redis-cli --rdb /backup/redis.rdb

# 备份配置文件
cp -r config/ /backup/config/
```

### 3. 性能调优

- 调整向量检索参数
- 优化缓存策略
- 调整并发处理能力
- 监控资源使用情况

## 故障排除

### 1. 常见问题

- **Milvus连接失败**：检查Milvus服务是否正常运行
- **Redis连接失败**：检查Redis服务和网络连接
- **内存不足**：调整批处理大小或增加内存
- **检索速度慢**：检查向量数据库索引配置

### 2. 日志分析

关键日志路径：
- 应用日志：`logs/app.log`
- 错误日志：`logs/error.log`
- 访问日志：`logs/access.log`

### 3. 性能分析

使用内置的性能指标监控系统状态，重点关注：
- QPS（每秒查询率）
- P99延迟
- 缓存命中率
- LLM调用成功率

## 企业级特性总结

1. **安全性**：JWT认证、RBAC权限、速率限制、数据加密
2. **可扩展性**：微服务架构、水平扩展、负载均衡
3. **高可用性**：健康检查、熔断机制、故障转移
4. **可观测性**：结构化日志、性能指标、错误追踪
5. **企业集成**：标准API、监控集成、审计日志
6. **性能优化**：多级缓存、异步处理、批量操作
7. **运维友好**：容器化部署、配置管理、自动化测试