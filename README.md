# astrbot_plugin_comfyui_workflow

一个 AstrBot 插件：触发 ComfyUI 工作流并返回生成图片。

## 指令

- `/<你配置的命令> <描述>`：按配置触发对应工作流并返回图片
- 群聊自然语言触发（可选）：以 `帮我画...` 等前缀开头的消息，会先用 AstrBot 当前会话的聊天模型生成绘画提示词，再执行绘图

管理指令：

- `/comfyui help`：查看当前可用命令
- `/comfyui refresh`：刷新配置页里的工作流文件下拉选项，并同步折叠标题

如果你发送 `/<命令>` 后仍然是大模型在回复（说明该平台/配置下 `/xxx` 未被插件拦截），请使用备用触发：

- `/comfyui run <命令> <提示词>`

插件会尝试把你配置的 `command/aliases` 注入为“指令别名”（让 `/<命令>` 能像普通指令一样被识别）。如果你的 AstrBot 版本/平台不支持动态别名注入，则 `/<命令>` 可能仍会被大模型接管，此时请使用 `/comfyui run`。

部分平台（例如 WebChat）可能会把你输入的 `/随机` 传给插件时变成 `随机`（去掉了斜杠）。插件已兼容直接发送 `随机` / `随机：提示词` / `随机 提示词` 作为触发。

如果你希望必须用 `/随机` 这类带斜杠的形式触发，请在配置中开启 `trigger_require_prefix`。开启后如果 WebChat 仍然会吞掉斜杠，你可以用 `//随机`（保留一个斜杠给插件）或改用 `/comfyui run 随机`。

注意：你在配置里填写的 `command` 是“动态命令”，不一定会出现在 AstrBot 的“指令管理/管理行为”列表中；请以 `/comfyui help` 输出为准。

## WebUI 配置

不要在配置页粘贴工作流 JSON。

1) 在 ComfyUI 中点击 Save -> API Format，导出 `.json` 文件
2) 推荐把导出的 `.json` 放到插件持久化目录：`data/plugin_data/astrbot_plugin_comfyui_workflow/workflows/`（更新插件不会丢）
   - 也兼容放到插件目录或插件目录的 `workflows/`，但更新插件可能会被覆盖/丢失
3) 在插件配置里新增一条或多条 `工作流` 配置，选择 `workflow_api_file` 并配置节点 ID 与命令

必填（最小可用配置）：

- `comfyui_base_url`：例如 `http://127.0.0.1:8188`
- `workflows` 里至少一条：
  - `command`：触发命令（用户输入 `/<command>`）
  - `workflow_api_file`：选择导出的 API JSON 文件
  - `output_node_id`：输出图片所在的节点 id（通常是 `SaveImage` 节点在 API JSON 里的顶层 key）

如果你希望把提示词写进工作流：

- `prompt_input_node_id`：正向提示词输入节点 id（通常是正向 `CLIPTextEncode` 节点，字段固定为 `text`）

可选：

- 在 `workflows` 里再加一条工作流，并把 `command` 设置为 `随机图`（或任意命令）
- `fixed_prompt`：固定提示词（用于随机图等无需参数的工作流）
- `require_prompt=false`：允许用户仅输入 `/<command>`
- `negative_prompt_input_node_id` / `default_negative_prompt`：把默认负面提示词与工作流内已有负面提示词拼接后写入（字段固定为 `text`）
