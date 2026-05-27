# ModelFeature 参数参考文档

> 模型 YAML 中 `features` 字段的完整说明，用于在 `models/llm/*.yaml` 中正确声明模型能力。

---

## 一、所有可用 Feature

| Feature | YAML 值 | 含义 | Dify 平台行为 |
|---------|---------|------|--------------|
| TOOL_CALL | `tool-call` | 支持单次工具调用 | 允许在 Agent/Workflow 中为该模型绑定工具 |
| MULTI_TOOL_CALL | `multi-tool-call` | 支持一次回复返回多个工具调用 | 模型可一次返回多个 tool_calls，而非逐个串行 |
| STREAM_TOOL_CALL | `stream-tool-call` | 支持流式工具调用 | 工具调用参数边生成边传递，减少首字延迟 |
| AGENT_THOUGHT | `agent-thought` | 支持 Agent 思维链 | Dify Agent 模式下显示思考过程 |
| VISION | `vision` | 支持图片输入 | Dify UI 显示上传图片入口，消息支持 ImagePromptMessage |
| DOCUMENT | `document` | 支持文档输入（PDF 等） | Dify UI 允许上传文档附件 |
| VIDEO | `video` | 支持视频输入 | Dify UI 允许上传视频附件 |
| AUDIO | `audio` | 支持音频输入 | Dify UI 允许上传音频附件 |
| STRUCTURED_OUTPUT | `structured-output` | 支持结构化输出 | Dify 可使用 JSON Schema 约束输出格式 |
| POLLING | `polling` | 支持轮询式异步调用 | 用于长耗时模型（如视频生成），Dify 通过轮询获取结果 |

**枚举定义位置**：`dify_plugin/entities/model/schema.py → class ModelFeature`

---

## 二、层级关系

工具调用相关的三个 feature 存在严格的层级关系：

```
tool-call                          ← 基础，必须先声明
  ├── multi-tool-call (子集)        ← 一次返回多个工具调用
  └── stream-tool-call (子集)       ← 流式传递工具调用参数
```

**规则**：

- 声明 `multi-tool-call` 或 `stream-tool-call` 时，**必须同时声明** `tool-call`
- 可以只声明 `tool-call`（表示只能返回单个工具调用）
- 官方仓库 908 个模型中，没有任何模型只有 `multi-tool-call` 而缺少 `tool-call`

---

## 三、DashScope 模型推荐配置

| 模型 | 类型 | 推荐 features |
|------|------|--------------|
| `qwen3.7-max` | 纯文本旗舰 | `tool-call`, `multi-tool-call`, `agent-thought`, `stream-tool-call` |
| `qwen3.6-plus` | 视觉多模态 | `vision`, `tool-call`, `multi-tool-call`, `agent-thought`, `stream-tool-call` |
| `qwen3.6-flash` | 视觉多模态（快速） | `vision`, `tool-call`, `multi-tool-call`, `agent-thought`, `stream-tool-call` |
| `qwen-plus` | 纯文本通用 | `tool-call`, `multi-tool-call`, `agent-thought`, `stream-tool-call` |
| `qwen-long` | 文档理解 | `document`, `agent-thought` |
| `qwen-vl-max` | 视觉专用 | `vision`, `agent-thought` |
| `qwen-omni-*` | 全模态 | `vision`, `video`, `audio`, `agent-thought` |

---

## 四、YAML 示例

### 纯文本模型

```yaml
features:
  - tool-call
  - multi-tool-call
  - agent-thought
  - stream-tool-call
```

### 视觉多模态模型

```yaml
features:
  - vision
  - tool-call
  - multi-tool-call
  - agent-thought
  - stream-tool-call
```

### 文档理解模型

```yaml
features:
  - document
  - agent-thought
```

---

## 五、Feature 与代码路由的关系

本插件的 `llm.py` 中，`ModelFeature.VISION` 用于判断 API 路由：

```python
if ModelFeature.VISION in (model_schema.features or []):
    # 使用 MultiModalConversation.call()
    response = MultiModalConversation.call(**params, ...)
else:
    # 使用 Generation.call()
    response = Generation.call(**params, ...)
```

因此，在模型 YAML 的 `features` 中添加或移除 `vision`，会直接影响该模型使用哪个 DashScope API。

---

## 六、官方仓库使用统计

| Feature | 使用数量 | 代表性模型 |
|---------|---------|-----------|
| `tool-call` | 908 | 几乎所有支持函数的模型 |
| `vision` | 811 | GPT-4o, Gemini, Claude 3.x, Qwen-VL |
| `multi-tool-call` | 811 | tool-call 的严格子集 |
| `stream-tool-call` | 598 | 大多数主流模型 |
| `audio` | ~60 | Gemini, GPT-4o-audio, Qwen3-Omni |
| `document` | 145 | Gemini, Claude 3.5+, Qwen-Long |
| `video` | 18 | Gemini 1.5/2.x, Mistral Pixtral |
| `structured-output` | 15 | Gemini 3.x, Claude Sonnet 4.6 |
| `polling` | 极少 | 异步长任务模型 |
