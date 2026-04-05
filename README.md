# 企业级私有知识库系统

## 项目概述

这是一个企业级的私有知识库问答系统，采用RAG（检索增强生成）架构，具备BM25缓存、混合检索、完整安全控制和微服务架构。

## 架构特性

### 1. 微服务架构
- **API网关**：统一入口、路由和负载均衡
- **认证服务**：JWT认证和RBAC权限控制
- **QA服务**：核心问答逻辑处理
- **向量存储服务**：支持ChromaDB/Milvus等
- **BM25缓存服务**：多级缓存机制
- **文档处理服务**：文档解析和索引

### 2. 安全特性
- **JWT认证**：安全的令牌管理
- **RBAC权限控制**：基于角色的访问控制
- **速率限制**：防止API滥用
- **熔断机制**：防止级联故障
- **数据加密**：敏感信息保护

### 3. 性能优化
- **多级缓存**：内存→Redis→持久化
- **异步处理**：高并发支持
- **向量化加速**：快速语义检索
- **混合检索**：向量+关键词结合

### 4. 可观测性
- **结构化日志**：统一日志格式
- **性能指标**：Prometheus监控
- **健康检查**：服务状态监控
- **错误追踪**：全面错误处理

## 部署方式

### 单体部署
```bash
# 安装依赖
pip install -e .

# 运行API服务器
python main.py server --host 0.0.0.0 --port 8000
```

### 微服务部署
```bash
# 使用Docker Compose部署所有服务
docker-compose up -d

# 或者单独运行各服务
python main.py microservice gateway --port 8000
python main.py microservice auth --port 8001
python main.py microservice qa --port 8002
python main.py microservice vector_store --port 8004
python main.py microservice bm25_cache --port 8005
```

## API端点

### 认证
- `POST /login` - 用户登录

### 问答功能
- `POST /query` - 查询接口（需认证）
- `POST /ingest` - 文档注入

### 系统监控
- `GET /health` - 健康检查
- `GET /metrics` - 性能指标

## 企业级功能

### 1. 安全控制
- 用户认证和授权
- 角色权限管理
- API访问控制
- 敏感数据保护

### 2. 容错机制
- 电路断路器
- 自动重试机制
- 降级策略
- 错误恢复

### 3. 可扩展性
- 水平扩展支持
- 负载均衡
- 服务发现
- 配置管理

### 4. 运维监控
- 日志管理
- 性能监控
- 告警机制
- 审计跟踪

## 技术栈

- **Web框架**: FastAPI
- **向量存储**: ChromaDB, Milvus
- **缓存**: Redis, 多级缓存
- **安全**: JWT, bcrypt
- **异步处理**: asyncio, aiohttp
- **监控**: Prometheus, structlog
- **容器化**: Docker, Docker Compose

## 开发规范

- 类型提示：完整的类型注解
- 错误处理：统一异常处理
- 日志记录：结构化日志
- 测试覆盖：单元测试和集成测试

## 维护说明

- 定期备份向量数据库
- 监控API性能指标
- 检查系统健康状态
- 更新安全证书和密钥

