import asyncio
import copy
from collections import deque
from array import array
from io import BytesIO
import json
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

_WORKFLOW_COMMAND_ALIASES: set[str] = set()


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


_SEP_CHARS = " \t\r\n：:，,"


def _starts_with_sep(s: str) -> bool:
    return bool(s) and s[0] in _SEP_CHARS


def _is_command_prefix(s: str) -> bool:
    if not s:
        return False
    return s.startswith("/") or s.startswith("／")


def _strip_command_prefix(s: str) -> str:
    if not s:
        return ""
    if s.startswith("/") or s.startswith("／"):
        return s[1:]
    return s


def _guess_raw_text_from_event(event: AstrMessageEvent) -> str | None:
    """Best-effort: try to recover original user input (may include leading '/')."""
    try:
        msg_obj = getattr(event, "message_obj", None)
        raw = getattr(msg_obj, "raw_message", None)
    except Exception:
        raw = None

    candidates: list[Any] = []
    if raw is not None:
        candidates.append(raw)

    # Some adapters may store original text at event.message_obj.raw_message.message
    if isinstance(raw, dict):
        for k in ("message_str", "message", "raw_message", "text", "content"):
            v = raw.get(k)
            if v is not None:
                candidates.append(v)

    for c in candidates:
        try:
            if isinstance(c, str):
                s = c.strip()
                if s:
                    return s
            if isinstance(c, dict):
                # Try one more level
                for k in ("text", "content", "message"):
                    v = c.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            continue

    return None


def _chunk_text(s: str, *, max_len: int = 1800) -> list[str]:
    s = s or ""
    if max_len <= 50:
        return [s]
    out: list[str] = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_len)
        out.append(s[start:end])
        start = end
    return out


def _chunks_count(s: str, *, max_len: int = 1800) -> int:
    if not s:
        return 1
    return max(1, (len(s) + max_len - 1) // max_len)


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
    negative_prompt_input_node_id: str | None
    default_negative_prompt: str | None
    seed_randomize: bool
    seed_input_node_id: str | None
    output_node_id: str | None
    output_image_index: int

    nlp_enable: bool
    nlp_require_at_bot: bool
    nlp_prefixes: tuple[str, ...]
    llm_prompt_instruction: str

    obfuscate_output: bool


@register(
    "astrbot_plugin_comfyui_workflow",
    "you",
    "对接 ComfyUI 工作流并返回图片",
    "0.2.2",
)
class ComfyUIWorkflowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None, *args, **kwargs):
        super().__init__(context)
        self.config = config or {}

        self._debug_recent_messages: deque[dict[str, Any]] = deque(maxlen=30)

        self._plugin_dir = Path(__file__).resolve().parent

        plugin_name = getattr(self, "name", None) or "astrbot_plugin_comfyui_workflow"
        data_root = _find_astrbot_data_dir_from_file(Path(__file__))
        self._plugin_data_dir = data_root / "plugin_data" / plugin_name
        self._images_dir = self._plugin_data_dir / "images"
        self._workflows_dir = self._plugin_data_dir / "workflows"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._workflows_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._sync_workflow_schema()
        except Exception:
            pass

        self._refresh_workflow_command_aliases()

    def _refresh_workflow_command_aliases(self) -> None:
        # This is a best-effort optimization:
        # If AstrBot keeps a reference to alias set, /<command> can be handled as a real command.
        # If AstrBot copies aliases at registration time, /comfyui run is still available.
        try:
            for w in self._get_workflow_specs():
                _WORKFLOW_COMMAND_ALIASES.add(w.command)
                _WORKFLOW_COMMAND_ALIASES.update(w.aliases)
        except Exception:
            pass

    @filter.command("comfyui_workflow", alias=_WORKFLOW_COMMAND_ALIASES, priority=100)
    async def comfyui_workflow_entry(self, event: AstrMessageEvent, words: str | None = None):
        # Note: AstrBot's command arg parser may treat varargs as required.
        # This command should be callable with no args.

        # Dispatch by parsing original message to know which workflow was invoked.
        msg = (event.message_str or "").strip()
        resolved = self._resolve_workflow_from_slash_message(msg)
        if resolved is None:
            yield event.plain_result(
                "这个指令入口用于工作流命令别名分发。\n"
                "请使用 /comfyui help 查看可用命令，或使用备用触发：/comfyui run <命令> <提示词>"
            )
            return

        workflow, user_prompt = resolved
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

    @comfyui.command("debug")
    async def comfyui_debug(self, event: AstrMessageEvent):
        if not bool(self.config.get("debug_enable", False)):
            yield event.plain_result("调试模式未开启：请在插件配置中开启 debug_enable 后再使用 /comfyui debug。")
            return

        try:
            config_obj = dict(self.config)
        except Exception:
            config_obj = {"__error__": "config is not dict-like"}

        try:
            workflows_raw = self.config.get("workflows")
        except Exception:
            workflows_raw = None

        specs = self._get_workflow_specs()
        discovered_files = []
        try:
            discovered_files = self._discover_workflow_api_files()
        except Exception:
            discovered_files = []

        raw_guess = _guess_raw_text_from_event(event)

        debug = {
            "plugin": {
                "name": getattr(self, "name", None) or "astrbot_plugin_comfyui_workflow",
                "version": "0.2.2",
                "plugin_dir": str(self._plugin_dir),
                "plugin_data_dir": str(self._plugin_data_dir),
            },
            "message": {
                "message_str": event.message_str,
                "raw_text_guess": raw_guess,
                "session_id": getattr(event, "session_id", None),
                "unified_msg_origin": getattr(event, "unified_msg_origin", None),
                "sender_id": (event.get_sender_id() if hasattr(event, "get_sender_id") else None),
                "sender_name": (event.get_sender_name() if hasattr(event, "get_sender_name") else None),
            },
            "config": config_obj,
            "workflows_raw": workflows_raw,
            "workflows_specs": [
                {
                    "name": w.name,
                    "command": w.command,
                    "aliases": sorted(w.aliases),
                    "workflow_api_file": w.workflow_api_file,
                    "output_node_id": w.output_node_id,
                    "prompt_input_node_id": w.prompt_input_node_id,
                    "negative_prompt_input_node_id": w.negative_prompt_input_node_id,
                    "seed_randomize": w.seed_randomize,
                    "seed_input_node_id": w.seed_input_node_id,
                    "nlp_enable": w.nlp_enable,
                    "nlp_require_at_bot": w.nlp_require_at_bot,
                    "nlp_prefixes": list(w.nlp_prefixes),
                    "llm_prompt_instruction_len": len(w.llm_prompt_instruction or ""),
                }
                for w in specs
            ],
            "runtime": {
                "workflow_command_aliases_count": len(_WORKFLOW_COMMAND_ALIASES),
                "workflow_command_aliases_sample": sorted(list(_WORKFLOW_COMMAND_ALIASES))[:80],
                "discovered_workflow_files": discovered_files,
                "recent_messages": list(self._debug_recent_messages),
            },
        }

        text = json.dumps(debug, ensure_ascii=False, indent=2)
        total = _chunks_count(text)
        for idx, chunk in enumerate(_chunk_text(text), start=1):
            prefix = f"[DEBUG {idx}/{total}]\n" if total > 1 else ""
            yield event.plain_result(prefix + chunk)

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

        self._refresh_workflow_command_aliases()
        if not files:
            yield event.plain_result("未发现任何 .json 工作流文件（插件目录或 workflows/）。")
            return
        shown = "\n".join(f"- {f}" for f in files[:30])
        suffix = "\n..." if len(files) > 30 else ""
        yield event.plain_result(f"已刷新工作流下拉选项，发现 {len(files)} 个文件：\n{shown}{suffix}\n请刷新 WebUI 配置页面。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def on_all_message(self, event: AstrMessageEvent):
        msg = (event.message_str or "").strip()

        raw_guess = None
        if bool(self.config.get("trigger_require_prefix", False)) or bool(self.config.get("debug_enable", False)):
            raw_guess = _guess_raw_text_from_event(event)

        if bool(self.config.get("debug_enable", False)):
            try:
                prefix = msg[:12]
                cps = [f"U+{ord(ch):04X}" for ch in prefix]
            except Exception:
                cps = []

            self._debug_recent_messages.append(
                {
                    "ts": int(time.time()),
                    "message_str": msg,
                    "raw_text_guess": raw_guess,
                    "prefix_codepoints": cps,
                }
            )

        resolved = self._resolve_workflow_from_message(msg, raw_guess=raw_guess)
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
        msg = (event.message_str or "").strip()
        workflows = [w for w in self._get_workflow_specs() if w.nlp_enable]
        if not workflows:
            return

        at_bot = self._is_at_bot(event)
        best: tuple[int, WorkflowSpec, str] | None = None
        for w in workflows:
            if w.nlp_require_at_bot and not at_bot:
                continue
            for p in w.nlp_prefixes:
                if not p:
                    continue
                if msg.startswith(p):
                    if best is None or len(p) > best[0]:
                        best = (len(p), w, p)

        if best is None:
            return

        _, workflow, prefix = best

        raw_req = msg[len(prefix) :].strip()
        raw_req = re.sub(r"^[：:，,\s]+", "", raw_req)
        if not raw_req:
            return

        event.stop_event()

        instruction = (workflow.llm_prompt_instruction or "").strip()
        prompt = ""
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            if provider_id:
                llm_prompt = f"{instruction}\n\n用户需求：{raw_req}" if instruction else raw_req
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

        async for result in self._run_workflow(event=event, workflow=workflow, prompt=prompt):
            yield result

    def _is_at_bot(self, event: AstrMessageEvent) -> bool:
        try:
            msg_obj = getattr(event, "message_obj", None)
            if msg_obj is None:
                return False
            self_id = str(getattr(msg_obj, "self_id", "") or "").strip()
            if not self_id:
                return False
            chain = getattr(msg_obj, "message", None)
            if not isinstance(chain, list):
                return False

            for seg in chain:
                try:
                    # At segment usually has 'qq'
                    qq = getattr(seg, "qq", None)
                    if qq is None:
                        continue
                    if str(qq).strip() == self_id:
                        return True
                except Exception:
                    continue

        except Exception:
            return False

        return False

    def _get_workflow_specs(self) -> list[WorkflowSpec]:
        raw = self.config.get("workflows")
        if not isinstance(raw, list):
            return []

        # Backward compatible global NLP settings (deprecated in schema)
        global_nlp_enable = bool(self.config.get("enable_nlp_draw", True))
        global_nlp_prefixes = self.config.get("nlp_draw_prefixes") or []
        if not isinstance(global_nlp_prefixes, list):
            global_nlp_prefixes = []
        global_nlp_prefixes = [str(p).strip() for p in global_nlp_prefixes if str(p).strip()]
        global_nlp_instruction = (self.config.get("llm_prompt_instruction") or "").strip()
        global_nlp_target = _normalize_cmd(self.config.get("nlp_target_command") or "")
        global_nlp_require_at = bool(self.config.get("nlp_require_at_bot", True))

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

            negative_prompt_input_node_id = (item.get("negative_prompt_input_node_id") or "").strip() or None
            default_negative_prompt = (item.get("default_negative_prompt") or "").strip() or None

            seed_randomize = bool(item.get("seed_randomize", False))
            seed_input_node_id = (item.get("seed_input_node_id") or "").strip() or None

            output_node_id = (item.get("output_node_id") or "").strip() or None
            output_image_index = int(item.get("output_image_index", 0) or 0)

            nlp_enable = bool(item.get("nlp_enable", False))
            nlp_require_at_bot = bool(item.get("nlp_require_at_bot", True))
            nlp_prefixes_raw = item.get("nlp_prefixes") or []
            if not isinstance(nlp_prefixes_raw, list):
                nlp_prefixes_raw = []
            nlp_prefixes = tuple(str(p).strip() for p in nlp_prefixes_raw if str(p).strip())
            llm_prompt_instruction = (item.get("llm_prompt_instruction") or "").strip()

            # Old behavior: enable NLP for a target command
            if ("nlp_enable" not in item) and global_nlp_enable and global_nlp_target and command == global_nlp_target:
                nlp_enable = True
            if ("nlp_prefixes" not in item) and global_nlp_prefixes:
                nlp_prefixes = tuple(global_nlp_prefixes)
            if ("llm_prompt_instruction" not in item) and global_nlp_instruction:
                llm_prompt_instruction = global_nlp_instruction
            if ("nlp_require_at_bot" not in item):
                nlp_require_at_bot = global_nlp_require_at

            obfuscate_output = bool(item.get("obfuscate_output", False))

            out.append(
                WorkflowSpec(
                    name=name,
                    command=command,
                    aliases=frozenset(aliases),
                    workflow_api_file=workflow_api_file,
                    require_prompt=require_prompt,
                    fixed_prompt=fixed_prompt,
                    prompt_input_node_id=prompt_input_node_id,
                    negative_prompt_input_node_id=negative_prompt_input_node_id,
                    default_negative_prompt=default_negative_prompt,
                    seed_randomize=seed_randomize,
                    seed_input_node_id=seed_input_node_id,
                    output_node_id=output_node_id,
                    output_image_index=output_image_index,

                    nlp_enable=nlp_enable,
                    nlp_require_at_bot=nlp_require_at_bot,
                    nlp_prefixes=nlp_prefixes,
                    llm_prompt_instruction=llm_prompt_instruction,

                    obfuscate_output=obfuscate_output,
                )
            )
        return out

    def _resolve_workflow_from_message(self, msg: str, *, raw_guess: str | None = None) -> tuple[WorkflowSpec, str] | None:
        # Supports:
        # - /<cmd> <prompt>
        # - ／<cmd> <prompt>
        # - <cmd> <prompt>     (for platforms that strip leading slash)
        if not msg:
            return None

        is_prefixed = _is_command_prefix(msg)
        if not is_prefixed and raw_guess:
            # If the adapter stripped the slash in message_str,
            # we still treat it as prefixed when raw text begins with a slash.
            is_prefixed = _is_command_prefix(raw_guess)

        if bool(self.config.get("trigger_require_prefix", False)) and not is_prefixed:
            return None

        rest = _strip_command_prefix(msg).lstrip() if is_prefixed else msg
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

            # If message is not explicitly prefixed, only accept formats like:
            #   随机
            #   随机 <prompt>
            #   随机：<prompt>
            # Avoid hijacking normal chat like "随机给我来一张".
            if not is_prefixed and remainder and not _starts_with_sep(remainder):
                continue

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
        # Prefer persisted directory first to survive plugin updates.
        search_dirs: list[Path] = [
            self._workflows_dir,
            self._plugin_dir / "workflows",
            self._plugin_dir,
        ]

        out: list[str] = []
        seen: set[str] = set()
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.glob("*.json"):
                if not p.is_file():
                    continue
                if p.name in {"_conf_schema.json"}:
                    continue
                name = p.name
                key = name.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(name)

        return sorted(out, key=lambda s: s.casefold())

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
        keep_template_keys: set[str] = {"workflow"}
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

                # Keep existing template key to avoid breaking UI state.
                template_key = str(entry.get("__template_key") or "").strip()
                if not template_key:
                    template_key = f"workflow_{entry_id}"
                    entry["__template_key"] = template_key
                    cfg_changed = True

                keep_template_keys.add(template_key)

                # Create/update the per-entry template so the collapse title shows entry_name.
                if template_key != "workflow":
                    tpl_meta = copy.deepcopy(base_tpl)
                    tpl_meta["name"] = entry_name
                    tpl_meta["hint"] = "已添加的工作流条目（用于显示标题）"
                    templates[template_key] = tpl_meta

        # Prune stale templates left by deleted entries.
        stale_keys = [
            k
            for k in list(templates.keys())
            if isinstance(k, str) and k.startswith("workflow_") and k not in keep_template_keys
        ]
        for k in stale_keys:
            try:
                del templates[k]
            except Exception:
                pass

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
        rel = (workflow_api_file or "").strip()
        if not rel:
            raise ValueError("工作流文件名为空")
        if not _is_safe_relpath(rel):
            raise ValueError(f"工作流文件路径不安全：{workflow_api_file}")

        # Allow selecting plain filenames; search persisted dir first.
        candidates = [
            (self._workflows_dir / rel),
            (self._plugin_dir / "workflows" / rel),
            (self._plugin_dir / rel),
        ]
        path = next((p for p in candidates if p.exists() and p.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"未找到工作流文件：{workflow_api_file}（已搜索：{', '.join(str(p) for p in candidates)}）"
            )

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

        if workflow.seed_randomize:
            # Use default ComfyUI-style seed range.
            # Follow comfyui_pro behavior: write seed/noise_seed for ALL nodes that have it.
            base_seed = secrets.randbelow(4294967296)
            seed_targets = self._auto_detect_seed_targets(workflow_json, prefer_node_id=workflow.seed_input_node_id)

            if not seed_targets:
                yield event.plain_result(
                    "已开启‘每次随机种子’，但未找到可写入 seed 的节点。\n"
                    "请在该工作流配置里填写 seed_input_node_id（通常为 KSampler 节点 ID）。"
                )
                return

            offset = 0
            for node_id, field in seed_targets:
                try:
                    self._set_workflow_input(
                        workflow_json,
                        node_id=node_id,
                        field=field,
                        value=int(base_seed + offset),
                    )
                    offset += 1
                except Exception as e:
                    yield event.plain_result(
                        f"写入随机 seed 失败：node_id={node_id} field={field}。请检查节点 ID/字段。错误：{e}"
                    )
                    return

        if prompt is not None and workflow.prompt_input_node_id:
            try:
                self._set_workflow_input(
                    workflow_json,
                    node_id=workflow.prompt_input_node_id,
                    field="text",
                    value=prompt,
                )
            except Exception as e:
                yield event.plain_result(f"写入提示词失败，请检查节点 ID/字段。错误：{e}")
                return

        if workflow.negative_prompt_input_node_id and workflow.default_negative_prompt:
            try:
                existing = self._get_workflow_input(
                    workflow_json,
                    node_id=workflow.negative_prompt_input_node_id,
                    field="text",
                )
                merged = self._merge_prompt_text(existing, workflow.default_negative_prompt)
                self._set_workflow_input(workflow_json, node_id=workflow.negative_prompt_input_node_id, field="text", value=merged)
            except Exception as e:
                yield event.plain_result(f"写入负面提示词失败，请检查节点 ID/字段。错误：{e}")
                return

        client = ComfyUIClient(comfyui_base_url)
        poll_interval = float(self.config.get("poll_interval_sec", 1.0))
        job_timeout = int(self.config.get("job_timeout_sec", 180))
        image_index = int(workflow.output_image_index)
        # Always pick first image; keep field only for backward compatibility.
        image_index = 0

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
            if workflow.obfuscate_output:
                try:
                    img_bytes = self._obfuscate_png_bytes(img_bytes)
                except Exception as e:
                    yield event.plain_result(f"图片混淆失败：{e}")
                    return
        except Exception as e:
            yield event.plain_result(f"生成失败: {e}")
            return

        out_path = self._write_image(img_bytes)
        self._gc_images(keep_last_n=int(self.config.get("save_last_n", 30)))
        if self._should_at_sender(event):
            sent = False
            try:
                from astrbot.api.message_components import At, Image

                sender_id = str(event.get_sender_id()) if hasattr(event, "get_sender_id") else ""
                if sender_id.isdigit():
                    chain = [At(qq=int(sender_id)), Image.fromFileSystem(str(out_path))]
                else:
                    chain = [Image.fromFileSystem(str(out_path))]
                yield event.chain_result(chain)
                sent = True
            except Exception:
                sent = False

            if sent:
                return

        yield event.image_result(str(out_path))

    def _should_at_sender(self, event: AstrMessageEvent) -> bool:
        if not bool(self.config.get("reply_at_sender", True)):
            return False
        try:
            msg_obj = getattr(event, "message_obj", None)
            group_id = getattr(msg_obj, "group_id", "") if msg_obj is not None else ""
            return bool(group_id)
        except Exception:
            return False

    @staticmethod
    def _set_workflow_input(workflow: dict[str, Any], *, node_id: str, field: str, value: Any) -> None:
        node = workflow.get(str(node_id))
        if not isinstance(node, dict):
            raise KeyError(f"未找到节点 id: {node_id}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise KeyError(f"节点 {node_id} 不包含 inputs 字段")
        inputs[field] = value

    @staticmethod
    def _get_workflow_input(workflow: dict[str, Any], *, node_id: str, field: str) -> Any:
        node = workflow.get(str(node_id))
        if not isinstance(node, dict):
            raise KeyError(f"未找到节点 id: {node_id}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise KeyError(f"节点 {node_id} 不包含 inputs 字段")
        return inputs.get(field)

    @staticmethod
    def _merge_prompt_text(existing: Any, add: str) -> str:
        add = (add or "").strip()
        if not add:
            return str(existing or "").strip()

        if existing is None:
            return add

        if isinstance(existing, str):
            left = existing.strip()
        else:
            left = str(existing).strip()

        if not left:
            return add

        # Simple concat; don't try to de-dup keywords.
        return f"{left}, {add}"

    @staticmethod
    def _auto_detect_seed_targets(
        workflow: dict[str, Any],
        *,
        prefer_node_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Find all nodes that accept seed/noise_seed.

        Order: preferred node first (if provided), then KSampler-like nodes, then others.
        """
        if not isinstance(workflow, dict):
            return []

        prefer_node_id = (prefer_node_id or "").strip() or None

        preferred: list[tuple[str, str]] = []
        ks: list[tuple[str, str]] = []
        other: list[tuple[str, str]] = []

        for node_id, node in workflow.items():
            if not isinstance(node_id, str):
                continue
            if not isinstance(node, dict):
                continue

            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue

            class_type = str(node.get("class_type") or "")
            class_cf = class_type.casefold()

            for k in ("seed", "noise_seed"):
                if k not in inputs:
                    continue
                target = (str(node_id), str(k))
                if prefer_node_id is not None and str(node_id) == prefer_node_id:
                    preferred.append(target)
                elif "ksampler" in class_cf:
                    ks.append(target)
                else:
                    other.append(target)

        combined = preferred + ks + other
        if not combined:
            return []

        seen = set()
        out: list[tuple[str, str]] = []
        for t in combined:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _gilbert2d_indices(width: int, height: int) -> array:
        """Generalized Hilbert ('gilbert') curve for width x height.

        Returns linear pixel indices (x + y*width) in visit order.
        """

        coords = array("I")

        def generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int) -> None:
            w = abs(ax + ay)
            h = abs(bx + by)

            dax = 0 if ax == 0 else (1 if ax > 0 else -1)
            day = 0 if ay == 0 else (1 if ay > 0 else -1)
            dbx = 0 if bx == 0 else (1 if bx > 0 else -1)
            dby = 0 if by == 0 else (1 if by > 0 else -1)

            if h == 1:
                for _ in range(w):
                    coords.append(x + y * width)
                    x += dax
                    y += day
                return

            if w == 1:
                for _ in range(h):
                    coords.append(x + y * width)
                    x += dbx
                    y += dby
                return

            ax2 = ax // 2
            ay2 = ay // 2
            bx2 = bx // 2
            by2 = by // 2

            w2 = abs(ax2 + ay2)
            h2 = abs(bx2 + by2)

            if 2 * w > 3 * h:
                if (w2 % 2) and (w > 2):
                    ax2 += dax
                    ay2 += day
                generate2d(x, y, ax2, ay2, bx, by)
                generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
            else:
                if (h2 % 2) and (h > 2):
                    bx2 += dbx
                    by2 += dby
                generate2d(x, y, bx2, by2, ax2, ay2)
                generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
                generate2d(
                    x + (ax - dax) + (bx2 - dbx),
                    y + (ay - day) + (by2 - dby),
                    -bx2,
                    -by2,
                    -(ax - ax2),
                    -(ay - ay2),
                )

        if width >= height:
            generate2d(0, 0, width, 0, 0, height)
        else:
            generate2d(0, 0, 0, height, width, 0)

        return coords

    @staticmethod
    def _obfuscate_png_bytes(png_bytes: bytes) -> bytes:
        """Obfuscate image pixels (reversible), like iead encryptImage()."""

        if Image is None:
            raise RuntimeError("缺少 Pillow，无法进行图片混淆")

        with Image.open(BytesIO(png_bytes)) as im:
            im = im.convert("RGBA")
            width, height = im.size
            if width <= 0 or height <= 0:
                raise RuntimeError("图片尺寸无效")

            src = im.tobytes()
            n = width * height
            curve = ComfyUIWorkflowPlugin._gilbert2d_indices(width, height)
            if len(curve) != n:
                raise RuntimeError("曲线长度与像素数量不一致")

            # offset = round(phi * n)
            offset = int(((5**0.5 - 1) / 2) * n + 0.5)

            src_mv = memoryview(src)
            dst = bytearray(len(src))
            dst_mv = memoryview(dst)

            for i in range(n):
                old_px = curve[i]
                new_px = curve[(i + offset) % n]
                op = old_px * 4
                np = new_px * 4
                dst_mv[np : np + 4] = src_mv[op : op + 4]

            out = Image.frombytes("RGBA", (width, height), bytes(dst))
            buf = BytesIO()
            out.save(buf, format="PNG")
            return buf.getvalue()

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
