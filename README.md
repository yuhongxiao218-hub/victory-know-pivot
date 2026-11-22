# Victory Know Pivot

面向设备说明书等资料场景的多模态 RAG Agent。项目将 PDF 和图片等资料加工为可检索知识库，并通过意图路由、任务规划、工具执行、证据评估与动态重规划，完成资料问答、设备查询和产品比较。

## 核心能力

- 文档导入：解析 PDF 及图片内容，清洗并切分文本，保留图文关联信息。
- 混合检索：结合向量召回、实体/设备名称确认和重排，提高检索相关性。
- Agent 查询：为复杂问题生成结构化任务计划，逐步调用检索工具并评估证据质量。
- 动态重规划：步骤失败或证据不足时调整计划，同时通过预算和重复检测避免无限循环。
- 可追踪输出：通过 SSE 返回执行进度，并记录计划、工具观察和证据评估轨迹。
- 会话能力：支持历史查询、上下文指代和需要补充信息时的澄清流程。

## 技术栈

- Python 3.11+
- FastAPI、Uvicorn
- LangChain、LangGraph、OpenAI-compatible API
- Milvus、MongoDB、MinIO
- BGE-M3、BGE Reranker
- MinerU

## 项目结构

```text
app/
├─ api/          # 导入与查询 HTTP 接口
├─ infra/        # 基础设施适配
├─ process/      # LangGraph 查询流程与 Agent 节点
├─ rag/          # 文档导入、检索、规划、执行和证据评估
├─ resources/    # 页面和 Prompt 资源
└─ shared/       # 配置、模型与通用工具
tests/           # 单元测试与集成测试
```

## 本地运行

1. 安装 [uv](https://docs.astral.sh/uv/) 并同步依赖：

   ```bash
   uv sync
   ```

2. 复制环境变量模板并填写模型及基础设施配置：

   ```bash
   cp .env.example .env
   ```

3. 启动资料导入服务：

   ```bash
   uv run uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8000
   ```

4. 启动查询服务：

   ```bash
   uv run uvicorn app.api.http.query_server:app --host 0.0.0.0 --port 8001
   ```

实际端口及外部服务地址以 `.env` 配置为准。查询服务提供健康检查、流式进度、查询、会话历史读取与清理接口。

## 测试

```bash
uv run pytest tests
```

