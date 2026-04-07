# 企业级集成问答系统 - 部署指南

## 项目概述

这是一个基于RAG（检索增强生成）的企业级问答系统，具备BM25缓存、混合检索、全面安全控制和微服务架构，采用Milvus作为主向量数据库。

## 项目架构

### 微服务架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   API Gateway   │───▶│ Authentication   │───▶│    QA Service    │
│   (Port 8000)   │    │   (Port 8001)    │    │   (Port 8002)    │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                             │                         │
                    ┌────────▼────────┐        ┌───────▼────────┐
            ┌──────▶│  BM25 Cache     │        │ Vector Store   │
            │       │   (Port 8005)   │        │  (Port 8004)   │
            │       └─────────────────┘        └────────────────┘
            │                                    (Milvus DB)
    ┌───────▼────────┐
    │Document Processor│
    │   (Port 8006)  │
    └────────────────┘
```

## 文件结构

### 核心模块

- `src/document_loader/` - 文档加载器（支持PDF、PPT、DOCX、图片）
- `src/document_processor/` - 文本处理和分块策略
- `src/vector_store/` - 向量嵌入和企业级向量存储（Milvus/ChromaDB）
- `src/bm25_cache/` - 增强BM25查询缓存层，支持多级缓存
- `src/dialogue_history/` - 多轮对话管理
- `src/rag_system/` - RAG组件：
  - `classifier.py` - 查询类型分类
  - `retriever.py` - 混合检索和重排序
  - `prompt_builder.py` - Prompt构建
- `src/security/` - 认证、授权和安全控制
  - `authentication.py` - JWT认证和RBAC
  - `middleware.py` - API保护安全中间件
  - `security_config.py` - 安全配置管理
- `src/utils/` - 工具函数，包括错误处理
  - `error_handling.py` - 熔断器、重试机制、限速控制
- `src/architecture/` - 微服务架构组件
  - `api_gateway.py` - API网关服务
  - `microservice_config.py` - 微服务配置
  - `service_orchestrator.py` - 服务编排层
- `src/api/` - FastAPI REST API服务器
- `src/cli/` - 命令行界面组件

### 主入口文件

- `main.py` - 单体应用主入口，支持命令行模式
- `microservice_main.py` - 微服务架构主入口
- `Dockerfile` - 应用容器化定义
- `docker-compose.yml` - 微服务编排配置（含Milvus集群）

## 部署方法

### 1. Docker Compose 部署（推荐）

创建 `.env` 文件：

```bash
JWT_SECRET_KEY=your-very-secret-key-change-this-in-production
OPENAI_API_KEY=your-openai-api-key-if-using-openai
```

启动所有服务：

```bash
docker-compose up -d
```

### 2. 单体应用部署

```bash
# 安装依赖
pip install -e .

# 启动单体服务器
python main.py server --host 0.0.0.0 --port 8000
```

### 3. 微服务部署

按顺序启动各服务：

```bash
# 启动API网关
python main.py microservice gateway --port 8000

# 启动认证服务
python main.py microservice auth --port 8001

# 启动QA服务
python main.py microservice qa --port 8002

# 启动向量存储服务（依赖Milvus）
python main.py microservice vector_store --port 8004

# 启动BM25缓存服务
python main.py microservice bm25_cache --port 8005
```

## 环境变量配置

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| JWT_SECRET_KEY | JWT密钥（必须设置） | - |
| LLM_API_KEY | LLM服务API密钥 | - |
| LLM_BASE_URL | LLM服务基础URL | http://localhost:11434/api |
| DOCUMENT_INGESTION_BASE_PATH | 允许的文档摄入基础路径 | ./data/documents |

## 使用方法

### 文档摄入

```bash
# 摄入单个文档
python main.py ingest path/to/document.pdf

# 摄入整个目录
python main.py ingest path/to/docs/
```

### 查询

```bash
# 查询
python main.py query "什么是 RAG?"
```

### API端点

- `GET /health` - 健康检查
- `POST /login` - 用户登录
- `POST /query` - 查询接口
- `POST /query/stream` - 流式查询
- `POST /ingest/file` - 文件上传
- `POST /profile` - 用户资料

## 安全特性

1. **JWT认证** - 所有API端点都需要认证
2. **RBAC权限控制** - 基于角色的访问控制
3. **速率限制** - Token Bucket算法
4. **路径遍历防护** - 防止恶意路径访问
5. **输入验证** - 所有输入都经过验证
6. **错误处理** - 熔断器和重试机制

## 企业级特性

1. **两阶段检索** - 首先BM25缓存，然后RAG处理未命中查询
2. **查询路由** - 通用查询→直接LLM；专业咨询→完整RAG管道
3. **混合搜索** - 向量相似度 + BM25关键词匹配
4. **模块化管道** - 每个组件可独立测试
5. **多级缓存** - 内存 → Redis → 持久化
6. **弹性模式** - 熔断器、重试、健康检查
7. **企业级向量存储** - 使用Milvus进行扩展性支持

## 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black .
ruff check --fix
```