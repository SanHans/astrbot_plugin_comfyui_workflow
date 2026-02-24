# astrbot_plugin_comfyui_workflow

一个 AstrBot 插件：触发 ComfyUI 工作流并返回生成图片。

## 指令

- `/<你配置的命令> <描述>`：按配置触发对应工作流并返回图片
- 群聊自然语言触发（可选）：以 `帮我画...` 等前缀开头的消息，会先用 AstrBot 当前会话的聊天模型生成绘画提示词，再执行绘图

管理指令：

- `/comfyui help`：查看当前可用命令
- `/comfyui refresh`：刷新配置页里的工作流文件下拉选项

## WebUI 配置

不要在配置页粘贴工作流 JSON。

1) 在 ComfyUI 中点击 Save -> API Format，导出 `.json` 文件
2) 把导出的 `.json` 放到插件目录下，或放到 `workflows/` 子目录下
3) 在插件配置里新增一条或多条 `工作流` 配置，选择 `workflow_api_file` 并配置节点 ID 与命令

必填（最小可用配置）：

- `comfyui_base_url`：例如 `http://127.0.0.1:8188`
- `workflows` 里至少一条：
  - `command`：触发命令（用户输入 `/<command>`）
  - `workflow_api_file`：选择导出的 API JSON 文件
  - `output_node_id`：输出图片所在的节点 id（通常是 `SaveImage` 节点在 API JSON 里的顶层 key）

如果你希望把提示词写进工作流：

- `prompt_input_node_id`：正向提示词输入节点 id（通常是正向 `CLIPTextEncode` 节点）
- `prompt_input_field`：字段名（通常是 `text`）

可选：

- 在 `workflows` 里再加一条工作流，并把 `command` 设置为 `随机图`（或任意命令）
- `fixed_prompt`：固定提示词（用于随机图等无需参数的工作流）
- `require_prompt=false`：允许用户仅输入 `/<command>`
- `negative_prompt_input_node_id` / `default_negative_prompt`：写入固定的负面提示词
