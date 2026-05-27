# Dify Plugin - DashScope Extended

基于官方 Tongyi 插件的增强版本，为阿里云百炼（DashScope）模型提供更精细的参数控制能力。

## 功能特性

相比官方插件，新增以下参数支持：

- **`enable_thinking`** - 深度思考模式开关（通用支持）
- **`search_options`** - 联网搜索高级选项（JSON 格式配置）
- **`parallel_tool_calls`** - 并行工具调用开关
- **`thinking_budget`** - 思考过程 token 预算限制

### 视觉模型支持

通过模型配置中的 `features` 字段声明 `vision` 能力，插件会自动路由到对应的 API：

- 包含 `vision` 的模型 → 使用 `MultiModalConversation` API
- 普通文本模型 → 使用 `Generation` API

## 安装

### 方式一：从 GitHub 仓库安装

```bash
# 克隆仓库
git clone https://github.com/Wiilz/dify-plugin-dashscope-extended.git

# 打包插件
cd dify-plugin-dashscope-extended
dify plugin package

# 上传生成的 .difypkg 文件到 Dify
```

### 方式二：直接下载

从 [Releases](https://github.com/Wiilz/dify-plugin-dashscope-extended/releases) 页面下载最新的 `.difypkg` 文件，然后在 Dify 控制台上传安装。

## 配置

### 1. 添加插件

在 Dify 控制台的 **插件市场** → **本地上传** 中上传 `.difypkg` 文件。

### 2. 配置凭证

安装后，在插件设置中填入：

- **DashScope API Key** - 从[阿里云百炼控制台](https://dashscope.console.aliyun.com/apiKey)获取

### 3. 选择模型

插件支持预定义模型和自定义模型两种方式：

**预定义模型**：
- qwen3.7-max
- qwen3.6-plus
- deepseek-v4-pro
- deepseek-v4-flash

**自定义模型**：
在 Dify 中添加自定义模型，填入模型名称即可。

## 参数说明

### 基础参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | float | 采样温度，控制随机性 |
| `top_p` | float | 核采样概率阈值 |
| `top_k` | int | 候选集大小 |
| `max_tokens` | int | 最大生成 token 数 |
| `seed` | int | 随机种子 |
| `repetition_penalty` | float | 重复惩罚系数 |

### 扩展参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `enable_thinking` | boolean | 开启深度思考模式，模型会先推理再回答 |
| `thinking_budget` | int | 思考过程的最大 token 数（仅当 `enable_thinking=true` 时生效） |
| `enable_search` | boolean | 开启联网搜索 |
| `search_options` | string | 联网搜索高级选项（JSON 字符串） |
| `parallel_tool_calls` | boolean | 允许并行调用多个工具 |
| `response_format` | string | 输出格式：`text` 或 `json_object` |

### search_options 配置示例

```json
{
  "forced_search": true,
  "search_strategy": "max",
  "assigned_site_list": ["baike.baidu.com", "zhihu.com"]
}
```

支持的选项：
- `enable_source` - 返回搜索来源
- `enable_citation` - 启用引用标注
- `forced_search` - 强制搜索
- `search_strategy` - 搜索策略
- `intention_options` - 意图选项
- `enable_search_extension` - 搜索扩展
- `assigned_site_list` - 指定搜索站点列表

## 添加新模型

编辑 `models/llm/` 目录下的 YAML 文件，或创建新的模型配置文件：

```yaml
model: your-model-name
label:
  en_US: Your Model Name
  zh_Hans: 你的模型名称
model_type: llm
features:
  - tool-call
  - multi-tool-call
  - agent-thought
  - stream-tool-call
  # 如果是视觉模型，添加：
  # - vision
model_properties:
  mode: chat
  context_size: 32768
parameter_rules:
  # 参数定义...
pricing:
  input: '0.00'
  output: '0.00'
  unit: '0.001'
  currency: RMB
```

详细配置说明参考 [docs/model-features.md](./docs/model-features.md)。

## 开发

### 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)（推荐的 Python 包管理器）
- Dify CLI（`dify` 命令）

### 本地开发

```bash
# 安装依赖
uv sync

# 本地运行插件（stdio 模式，带日志）
dify plugin run . -l

# 本地运行插件（TCP 模式）
dify plugin run . -m tcp
```

### 打包

```bash
dify plugin package
```

生成的 `.difypkg` 文件可上传到 Dify 安装。

## 日志查看

插件使用 `plugin_logger_handler` 输出日志，在社区版 Dify 中可通过以下方式查看：

```bash
docker logs -f dify-plugin-daemon | grep dashscope-extended
```

日志示例：
```
[dashscope-extended] _generate called: model=qwen3.7-max, model_parameters=['temperature', 'enable_search', 'search_options', ...], tools=0, stream=True
[dashscope-extended] Generation 请求参数: {"model": "qwen3.7-max", "temperature": 0.3, "enable_search": true, ...}
```

## 项目结构

```
dashscope-extended/
├── manifest.yaml              # 插件清单
├── main.py                    # 插件入口
├── pyproject.toml             # 依赖配置
├── provider/
│   ├── dashscope-extended.yaml  # Provider 配置
│   └── dashscope-extended.py    # Provider 实现
├── models/
│   ├── _common.py             # 公共工具
│   └── llm/
│       ├── llm.py             # LLM 核心实现
│       ├── _position.yaml     # 模型排序
│       ├── qwen3.7-max.yaml   # 模型配置
│       └── ...
└── docs/
    └── model-features.md      # 模型特性文档
```

## 与官方插件的区别

| 特性 | 官方 Tongyi 插件 | 本插件 |
|------|------------------|--------|
| `enable_thinking` | 仅部分模型支持 | 通用支持 |
| `search_options` | ❌ | ✅ JSON 配置 |
| `parallel_tool_calls` | ❌ | ✅ |
| 视觉模型路由 | 基于 features | 基于 features |
| 流式输出 | 动态判断 | 固定启用 |

## 许可证

本项目遵循与 Dify 官方插件相同的许可证。

## 相关链接

- [Dify 官方文档](https://docs.dify.ai)
- [DashScope API 文档](https://help.aliyun.com/zh/dashscope/)
- [Dify 插件开发指南](https://docs.dify.ai/plugins/quick-start)
