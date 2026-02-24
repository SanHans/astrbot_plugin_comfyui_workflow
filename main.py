import asyncio
import copy
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


def _find_astrbot_data_dir_from_file(file_path: Path) -> Path:
    curr = file_path.resolve()
    if curr.is_file():
        curr = curr.parent

    while True:
        if curr.name.lower() == "data":
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent

    return (Path.cwd() / "data").resolve()


def _normalize_cmd(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("/"):
        s = s[1:]
    return s.strip().casefold()


def _is_safe_relpath(relpath: str) -> bool:
    if not relpath:
        return False
    p = Path(relpath)
    if p.is_absolute():
        return False
    if any(part in {"..", ""} for part in p.parts):
        return False
    return True


def _strip_prompt_separators(s: str) -> str:
    return re.sub(r"^[\s：:，,]+", "", (s or "").strip())


@dataclass(frozen=True)
class ComfyUIImageRef:
    filename: str
    subfolder: str | None = None
    type: str | None = None


class ComfyUIClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())

    async def queue_prompt(self, workflow: dict[str, Any], timeout_sec: float = 30) -> str:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout_sec) as client:
            resp = await client.post(
                "/prompt",
                json={"prompt": workflow, "client_id": self._client_id},
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
            return str(prompt_id)

    async def get_history(self, prompt_id: str, timeout_sec: float = 30) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout_sec) as client:
            resp = await client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            return resp.json()

    async def view_image(self, ref: ComfyUIImageRef, timeout_sec: float = 60) -> bytes:
        params: dict[str, str] = {"filename": ref.filename}
        if ref.subfolder:
            params["subfolder"] = ref.subfolder
        if ref.type:
            params["type"] = ref.type

        async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout_sec) as client:
            resp = await client.get("/view", params=params)
            resp.raise_for_status()
            return resp.content


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    command: str
    aliases: frozenset[str]
    workflow_api_file: str | None
    require_prompt: bool
    fixed_prompt: str | None
    prompt_input_node_id: str | None
    prompt_input_field: str
    negative_prompt_input_node_id: str | None
    negative_prompt_input_field: str
    default_negative_prompt: str | None
    output_node_id: str | None
    output_image_index: int


@register(
    "astrbot_plugin_comfyui_workflow",
    "you",
    "对接 ComfyUI 工作流并返回图片",
    "0.2.0",
)
class ComfyUIWorkflowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None, *args, **kwargs):
        super().__init__(context)
        self.config = config or {}

        self._plugin_dir = Path(__file__).resolve().parent

        plugin_name = getattr(self, "name", None) or "astrbot_plugin_comfyui_workflow"
        data_root = _find_astrbot_data_dir_from_file(Path(__file__))
        self._plugin_data_dir = data_root / "plugin_data" / plugin_name
        self._images_dir = self._plugin_data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._sync_workflow_schema()
        except Exception:
            pass

    @filter.command_group("comfyui", alias={"comfy"})
    def comfyui(self):
        pass

    @comfyui.command("help")
    async def comfyui_help(self, event: AstrMessageEvent):
        workflows = self._get_workflow_specs()
        if not workflows:
            yield event.plain_result(
                "尚未配置任何工作流。\n"
                "1) 把 ComfyUI 导出的 API JSON 放到插件目录或 workflows/\n"
                "2) 在 WebUI 插件配置里新增一条‘工作流’配置\n"
                "3) 选择文件并填写 output_node_id\n"
                "新增文件后可执行：/comfyui refresh"
            )
            return

        lines = ["可用工作流命令："]
        for w in workflows:
            aliases = sorted(a for a in w.aliases if a != w.command)
            alias_text = f"（别名：{', '.join(aliases)}）" if aliases else ""
            problems: list[str] = []
            if not w.workflow_api_file:
                problems.append("未选择工作流文件")
            if not w.output_node_id:
                problems.append("未填写 output_node_id")
            suffix = f"（{', '.join(problems)}）" if problems else ""
            lines.append(f"- /{w.command} {alias_text}{suffix}".rstrip())
        lines.append("\n管理指令：/comfyui refresh")
        lines.append("备用触发：/comfyui run <命令> <提示词>")
        yield event.plain_result("\n".join(lines))

    @comfyui.command("run")
    async def comfyui_run(self, event: AstrMessageEvent, command: str | None = None, *words: str):
        cmd = _normalize_cmd(command or "")
        if not cmd:
            yield event.plain_result("用法：/comfyui run <命令> <提示词>\n示例：/comfyui run 画图 一只戴墨镜的橘猫")
            return

        workflow = self._find_workflow_by_command(cmd)
        if workflow is None:
            yield event.plain_result(f"未找到命令对应的工作流：{command}")
            return

        user_prompt = " ".join(words).strip()
        prompt: str | None
        if workflow.fixed_prompt:
            prompt = workflow.fixed_prompt
        else:
            prompt = user_prompt or None

        if workflow.require_prompt and not prompt:
            yield event.plain_result(f"用法：/{workflow.command} 你的描述")
            return

        yield event.plain_result("已收到绘图请求，正在生成，请稍等...")
        async for result in self._run_workflow(event=event, workflow=workflow, prompt=prompt):
            yield result

    @comfyui.command("refresh")
    async def comfyui_refresh(self, event: AstrMessageEvent):
        try:
            files = self._sync_workflow_schema()
        except Exception as e:
            yield event.plain_result(f"刷新失败：{e}")
            return
        if not files:
            yield event.plain_result("未发现任何 .json 工作流文件（插件目录或 workflows/）。")
            return
        shown = "\n".join(f"- {f}" for f in files[:30])
        suffix = "\n..." if len(files) > 30 else ""
        yield event.plain_result(f"已刷新工作流下拉选项，发现 {len(files)} 个文件：\n{shown}{suffix}\n请刷新 WebUI 配置页面。")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        msg = (event.message_str or "").strip()
        if not msg.startswith("/"):
            return

        resolved = self._resolve_workflow_from_slash_message(msg)
        if resolved is None:
            return

        workflow, user_prompt = resolved
        event.stop_event()

        prompt: str | None
        if workflow.fixed_prompt:
            prompt = workflow.fixed_prompt
        else:
            prompt = user_prompt or None

        if workflow.require_prompt and not prompt:
            yield event.plain_result(f"用法：/{workflow.command} 你的描述")
            return

        yield event.plain_result("已收到绘图请求，正在生成，请稍等...")
        async for result in self._run_workflow(event=event, workflow=workflow, prompt=prompt):
            yield result

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def nlp_draw(self, event: AstrMessageEvent):
        if not bool(self.config.get("enable_nlp_draw", True)):
            return

        msg = (event.message_str or "").strip()
        prefixes: list[str] = self.config.get("nlp_draw_prefixes") or []
        prefixes = [p.strip() for p in prefixes if (p or "").strip()]
        if not prefixes:
            return

        prefix = next((p for p in prefixes if msg.startswith(p)), None)
        if not prefix:
            return

        raw_req = msg[len(prefix) :].strip()
        raw_req = re.sub(r"^[：:，,\s]+", "", raw_req)
        if not raw_req:
            return

        event.stop_event()

        instruction = (self.config.get("llm_prompt_instruction") or "").strip()
        prompt = ""
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            if provider_id:
                llm_prompt = f"{instruction}\n\n用户需求：{raw_req}"
                llm_resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=llm_prompt)
                prompt = (llm_resp.completion_text or "").strip()
                prompt = re.sub(r"^```.*?\n|```$", "", prompt, flags=re.DOTALL).strip()
        except Exception:
            prompt = ""

        if not prompt:
            prompt = raw_req
            yield event.plain_result("未能获取 AI 润色提示词，已使用原描述继续绘图。")
        else:
            yield event.plain_result(f"已为你整理提示词并开始绘制：{prompt}")

        target_cmd = _normalize_cmd(self.config.get("nlp_target_command") or "")
        workflow = self._find_workflow_by_command(target_cmd) if target_cmd else None
        if workflow is None:
            workflow = self._get_workflow_specs()[0] if self._get_workflow_specs() else None
        if workflow is None:
            yield event.plain_result("尚未配置任何工作流，无法执行绘图。")
            return

        async for result in self._run_workflow(event=event, workflow=workflow, prompt=prompt):
            yield result

    def _get_workflow_specs(self) -> list[WorkflowSpec]:
        raw = self.config.get("workflows")
        if not isinstance(raw, list):
            return []

        out: list[WorkflowSpec] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            name = (item.get("name") or "").strip() or "工作流"
            command = _normalize_cmd(item.get("command") or "")
            if not command:
                continue

            aliases_raw = item.get("aliases")
            aliases: set[str] = set()
            if isinstance(aliases_raw, list):
                for a in aliases_raw:
                    a_norm = _normalize_cmd(str(a))
                    if a_norm:
                        aliases.add(a_norm)
            aliases.add(command)

            workflow_api_file = (item.get("workflow_api_file") or "").strip() or None

            require_prompt = bool(item.get("require_prompt", True))
            fixed_prompt = (item.get("fixed_prompt") or "").strip() or None

            prompt_input_node_id = (item.get("prompt_input_node_id") or "").strip() or None
            prompt_input_field = (item.get("prompt_input_field") or "text").strip() or "text"

            negative_prompt_input_node_id = (item.get("negative_prompt_input_node_id") or "").strip() or None
            negative_prompt_input_field = (item.get("negative_prompt_input_field") or "text").strip() or "text"
            default_negative_prompt = (item.get("default_negative_prompt") or "").strip() or None

            output_node_id = (item.get("output_node_id") or "").strip() or None
            output_image_index = int(item.get("output_image_index", 0) or 0)

            out.append(
                WorkflowSpec(
                    name=name,
                    command=command,
                    aliases=frozenset(aliases),
                    workflow_api_file=workflow_api_file,
                    require_prompt=require_prompt,
                    fixed_prompt=fixed_prompt,
                    prompt_input_node_id=prompt_input_node_id,
                    prompt_input_field=prompt_input_field,
                    negative_prompt_input_node_id=negative_prompt_input_node_id,
                    negative_prompt_input_field=negative_prompt_input_field,
                    default_negative_prompt=default_negative_prompt,
                    output_node_id=output_node_id,
                    output_image_index=output_image_index,
                )
            )
        return out

    def _resolve_workflow_from_slash_message(self, msg: str) -> tuple[WorkflowSpec, str] | None:
        if not msg.startswith("/"):
            return None
        rest = msg[1:].lstrip()
        if not rest:
            return None

        workflows = self._get_workflow_specs()
        if not workflows:
            return None

        # First try: whitespace-delimited command
        parts = rest.split(maxsplit=1)
        cmd = _normalize_cmd(parts[0])
        wf = self._find_workflow_by_command(cmd)
        if wf is not None:
            prompt = _strip_prompt_separators(parts[1] if len(parts) > 1 else "")
            return wf, prompt

        # Second try: allow no-space usage like /画图xxx or /画图：xxx
        # Prefer longest match to avoid prefix collisions.
        alias_pairs: list[tuple[str, WorkflowSpec]] = []
        for w in workflows:
            for a in w.aliases:
                alias_pairs.append((a, w))
        alias_pairs.sort(key=lambda x: len(x[0]), reverse=True)

        rest_cf = rest.casefold()
        for alias, w in alias_pairs:
            if not alias:
                continue
            if not rest_cf.startswith(alias):
                continue
            remainder = rest[len(alias) :]
            prompt = _strip_prompt_separators(remainder)
            return w, prompt

        return None

    def _find_workflow_by_command(self, cmd: str) -> WorkflowSpec | None:
        cmd_norm = _normalize_cmd(cmd)
        if not cmd_norm:
            return None
        for w in self._get_workflow_specs():
            if cmd_norm in w.aliases:
                return w
        return None

    def _discover_workflow_api_files(self) -> list[str]:
        candidates: list[Path] = []
        candidates.extend(self._plugin_dir.glob("*.json"))
        candidates.extend((self._plugin_dir / "workflows").glob("*.json"))

        out: list[str] = []
        for p in candidates:
            if not p.is_file():
                continue
            if p.name in {"_conf_schema.json"}:
                continue
            try:
                rel = p.relative_to(self._plugin_dir)
            except ValueError:
                continue
            out.append(rel.as_posix())

        out = sorted(set(out), key=lambda s: s.casefold())
        return out

    def _sync_workflow_schema(self) -> list[str]:
        files = self._discover_workflow_api_files()
        schema_path = self._plugin_dir / "_conf_schema.json"
        if not schema_path.exists():
            return files

        schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
        workflows_meta = schema.get("workflows")
        if not isinstance(workflows_meta, dict):
            return files

        templates = workflows_meta.get("templates")
        if not isinstance(templates, dict):
            templates = {}
            workflows_meta["templates"] = templates

        base_tpl = templates.get("workflow")
        if not isinstance(base_tpl, dict):
            return files
        base_items = base_tpl.get("items")
        if not isinstance(base_items, dict):
            return files
        wf_file = base_items.get("workflow_api_file")
        if isinstance(wf_file, dict):
            wf_file["options"] = files

        cfg_changed = False
        cfg_workflows = self.config.get("workflows")
        if isinstance(cfg_workflows, list):
            for entry in cfg_workflows:
                if not isinstance(entry, dict):
                    continue

                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    entry_id = uuid.uuid4().hex
                    entry["id"] = entry_id
                    cfg_changed = True

                entry_name = str(entry.get("name") or "").strip() or "工作流"
                template_key = f"workflow_{entry_id}"
                if entry.get("__template_key") != template_key:
                    entry["__template_key"] = template_key
                    cfg_changed = True

                tpl_meta = copy.deepcopy(base_tpl)
                tpl_meta["name"] = entry_name
                tpl_meta["hint"] = "已添加的工作流条目（用于显示标题）"
                templates[template_key] = tpl_meta

        schema_path.write_text(json.dumps(schema, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        if cfg_changed:
            save = getattr(self.config, "save_config", None)
            if callable(save):
                try:
                    save()
                except Exception:
                    pass

        return files

    def _load_workflow_api_json(self, workflow_api_file: str) -> dict[str, Any]:
        rel = workflow_api_file.strip()
        if not _is_safe_relpath(rel):
            raise ValueError(f"工作流文件路径不安全：{workflow_api_file}")
        path = (self._plugin_dir / rel).resolve()
        if self._plugin_dir not in path.parents and path != self._plugin_dir:
            raise ValueError(f"工作流文件不在插件目录内：{workflow_api_file}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"未找到工作流文件：{workflow_api_file}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"工作流文件不是有效的 API JSON 对象：{workflow_api_file}")
        return data

    async def _run_workflow(self, *, event: AstrMessageEvent, workflow: WorkflowSpec, prompt: str | None):
        comfyui_base_url = (self.config.get("comfyui_base_url") or "").strip()
        if not comfyui_base_url:
            yield event.plain_result("插件未配置 comfyui_base_url，请先在插件配置中填写 ComfyUI 地址。")
            return

        if not workflow.workflow_api_file:
            yield event.plain_result(f"工作流『{workflow.name}』未选择 workflow_api_file，请先在插件配置里选择工作流文件。")
            return

        if not workflow.output_node_id:
            yield event.plain_result(f"工作流『{workflow.name}』未填写 output_node_id，请先在插件配置里填写输出节点 ID（通常是 SaveImage 节点）。")
            return

        try:
            workflow_json = self._load_workflow_api_json(workflow.workflow_api_file)
        except Exception as e:
            yield event.plain_result(f"读取工作流文件失败：{e}")
            return

        if prompt is not None and workflow.prompt_input_node_id:
            try:
                self._set_workflow_input(
                    workflow_json,
                    node_id=workflow.prompt_input_node_id,
                    field=workflow.prompt_input_field,
                    value=prompt,
                )
            except Exception as e:
                yield event.plain_result(f"写入提示词失败，请检查节点 ID/字段。错误：{e}")
                return

        if workflow.negative_prompt_input_node_id and workflow.default_negative_prompt:
            try:
                self._set_workflow_input(
                    workflow_json,
                    node_id=workflow.negative_prompt_input_node_id,
                    field=workflow.negative_prompt_input_field,
                    value=workflow.default_negative_prompt,
                )
            except Exception as e:
                yield event.plain_result(f"写入负面提示词失败，请检查节点 ID/字段。错误：{e}")
                return

        client = ComfyUIClient(comfyui_base_url)
        poll_interval = float(self.config.get("poll_interval_sec", 1.0))
        job_timeout = int(self.config.get("job_timeout_sec", 180))
        image_index = int(workflow.output_image_index)

        try:
            prompt_id = await client.queue_prompt(workflow_json)
            img_ref = await self._wait_for_output_image(
                client=client,
                prompt_id=prompt_id,
                output_node_id=workflow.output_node_id,
                image_index=image_index,
                poll_interval_sec=poll_interval,
                timeout_sec=job_timeout,
            )
            img_bytes = await client.view_image(img_ref)
        except Exception as e:
            yield event.plain_result(f"生成失败: {e}")
            return

        out_path = self._write_image(img_bytes)
        self._gc_images(keep_last_n=int(self.config.get("save_last_n", 30)))
        yield event.image_result(str(out_path))

    @staticmethod
    def _set_workflow_input(workflow: dict[str, Any], *, node_id: str, field: str, value: Any) -> None:
        node = workflow.get(str(node_id))
        if not isinstance(node, dict):
            raise KeyError(f"未找到节点 id: {node_id}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise KeyError(f"节点 {node_id} 不包含 inputs 字段")
        inputs[field] = value

    async def _wait_for_output_image(
        self,
        *,
        client: ComfyUIClient,
        prompt_id: str,
        output_node_id: str,
        image_index: int,
        poll_interval_sec: float,
        timeout_sec: int,
    ) -> ComfyUIImageRef:
        deadline = time.time() + max(1, timeout_sec)
        last_err: str | None = None

        while time.time() < deadline:
            hist = await client.get_history(prompt_id)
            prompt_hist = hist.get(prompt_id)
            if isinstance(prompt_hist, dict):
                outputs = prompt_hist.get("outputs")
                if isinstance(outputs, dict):
                    out = outputs.get(str(output_node_id))
                    if isinstance(out, dict):
                        images = out.get("images")
                        if isinstance(images, list) and images:
                            idx = max(0, image_index)
                            if idx < len(images) and isinstance(images[idx], dict):
                                img = images[idx]
                                filename = img.get("filename")
                                if filename:
                                    return ComfyUIImageRef(
                                        filename=str(filename),
                                        subfolder=img.get("subfolder") or None,
                                        type=img.get("type") or None,
                                    )
                            last_err = f"输出图片序号无效：{image_index}"
                        else:
                            last_err = "输出节点暂未产出图片"
                    else:
                        last_err = f"输出节点尚未就绪：{output_node_id}"
                else:
                    last_err = "工作流输出尚未就绪"
            else:
                last_err = "任务历史尚未就绪"

            await asyncio.sleep(max(0.2, poll_interval_sec))

        raise TimeoutError(last_err or "任务超时")

    def _write_image(self, img_bytes: bytes) -> Path:
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        path = self._images_dir / name
        path.write_bytes(img_bytes)
        return path

    def _gc_images(self, *, keep_last_n: int) -> None:
        keep_last_n = max(0, int(keep_last_n))
        if keep_last_n == 0:
            return
        files = sorted(self._images_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep_last_n:]:
            try:
                os.remove(p)
            except OSError:
                pass
