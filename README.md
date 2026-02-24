# astrbot_plugin_comfyui_workflow

一个 AstrBot 插件：触发 ComfyUI 工作流并返回生成图片。

## 指令

- `/随机图`：执行“随机图”工作流（未配置则复用主工作流）并返回图片
- `/画图 <描述>`：将 `<描述>` 写入配置的输入节点后执行工作流并返回图片
- 群聊自然语言触发（可选）：以 `帮我画...` 等前缀开头的消息，会先用 AstrBot 当前会话的聊天模型生成绘画提示词，再执行绘图

## WebUI 配置

把你在 ComfyUI 里导出的 “API format” 工作流 JSON 粘贴到 `workflow_api_json`。

必填（最小可用配置）：

- `comfyui_base_url`：例如 `http://127.0.0.1:8188`
- `output_node_id`：输出图片所在的节点 id（通常是 `SaveImage` 节点在 API JSON 里的顶层 key）

如果你希望 `/画图` 能把提示词写进工作流：

- `prompt_input_node_id`：正向提示词输入节点 id（通常是正向 `CLIPTextEncode` 节点）
- `prompt_input_field`：字段名（通常是 `text`）

可选：

- `random_workflow_api_json` / `random_output_node_id` / `random_output_image_index`：单独给 `/随机图` 指定另一套工作流/输出
- `negative_prompt_input_node_id` / `default_negative_prompt`：写入固定的负面提示词
