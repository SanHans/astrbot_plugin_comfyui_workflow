import asyncio
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


@register(
    "astrbot_plugin_comfyui_workflow",
    "you",
    "对接 ComfyUI 工作流并返回图片",
    "0.1.1",
)
class ComfyUIWorkflowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None, *args, **kwargs):
        super().__init__(context)
        self.config = config or {}

        plugin_name = getattr(self, "name", None) or "astrbot_plugin_comfyui_workflow"
        data_root = _find_astrbot_data_dir_from_file(Path(__file__))
        self._plugin_data_dir = data_root / "plugin_data" / plugin_name
        self._images_dir = self._plugin_data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

    @filter.command("随机图", alias={"random图", "随机"})
    async def random_image(self, event: AstrMessageEvent):
        prompt = (self.config.get("random_prompt") or "").strip()
        yield event.plain_result("已收到随机绘图请求，正在生成，请稍等...")
        random_workflow = (self.config.get("random_workflow_api_json") or "").strip() or None
        random_output_node_id = (self.config.get("random_output_node_id") or "").strip() or None
        random_output_index = int(self.config.get("random_output_image_index", -1))
        async for result in self._run_and_build_results(
            event=event,
            prompt=prompt or None,
            workflow_api_json_override=random_workflow,
            output_node_id_override=random_output_node_id,
            output_image_index_override=(None if random_output_index < 0 else random_output_index),
        ):
            yield result

    @filter.command("画图", alias={"draw"})
    async def draw(self, event: AstrMessageEvent, *words: str):
        prompt = " ".join(words).strip()
        if not prompt:
            yield event.plain_result("用法：/画图 你的描述\n示例：/画图 一只戴墨镜的橘猫，电影感，4k")
            return
        yield event.plain_result("已收到绘图请求，正在生成，请稍等...")
        async for result in self._run_and_build_results(event=event, prompt=prompt):
            yield result

    @filter.command("绘图帮助", alias={"画图帮助", "comfy帮助", "comfyui帮助"})
    async def draw_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "可用指令：\n"
            "1) /画图 你的描述\n"
            "2) /随机图\n"
            "3) 群聊发送“帮我画xxx”\n\n"
            "首次使用请在插件配置里至少填写：\n"
            "- comfyui_base_url\n"
            "- workflow_api_json\n"
            "- output_node_id\n"
            "如果 /画图 不生效，再补充 prompt_input_node_id 与 prompt_input_field。"
        )

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

        async for result in self._run_and_build_results(event=event, prompt=prompt):
            yield result

    async def _run_and_build_results(
        self,
        *,
        event: AstrMessageEvent,
        prompt: str | None,
        workflow_api_json_override: str | None = None,
        output_node_id_override: str | None = None,
        output_image_index_override: int | None = None,
    ):
        comfyui_base_url = (self.config.get("comfyui_base_url") or "").strip()
        if not comfyui_base_url:
            yield event.plain_result("插件未配置 comfyui_base_url，请先在插件配置中填写 ComfyUI 地址。")
            return

        workflow_api_json = (workflow_api_json_override or (self.config.get("workflow_api_json") or "{}")).strip()
        try:
            workflow: dict[str, Any] = json.loads(workflow_api_json)
        except json.JSONDecodeError as e:
            yield event.plain_result(f"workflow_api_json 不是有效 JSON，请检查格式。错误：{e}")
            return

        output_node_id = (output_node_id_override or (self.config.get("output_node_id") or "")).strip()
        if not output_node_id:
            yield event.plain_result("插件未配置 output_node_id，请填写输出图片节点 id（通常是 SaveImage 节点）。")
            return

        prompt_node_id = (self.config.get("prompt_input_node_id") or "").strip()
        prompt_field = (self.config.get("prompt_input_field") or "text").strip() or "text"

        neg_node_id = (self.config.get("negative_prompt_input_node_id") or "").strip()
        neg_field = (self.config.get("negative_prompt_input_field") or "text").strip() or "text"
        default_negative = (self.config.get("default_negative_prompt") or "").strip()

        if prompt is not None and prompt_node_id:
            try:
                self._set_workflow_input(workflow, node_id=prompt_node_id, field=prompt_field, value=prompt)
            except Exception as e:
                yield event.plain_result(f"写入提示词失败，请检查 prompt_input_node_id / prompt_input_field。错误：{e}")
                return

        if neg_node_id and default_negative:
            try:
                self._set_workflow_input(workflow, node_id=neg_node_id, field=neg_field, value=default_negative)
            except Exception as e:
                yield event.plain_result(f"写入负面提示词失败，请检查 negative_prompt_input_node_id / negative_prompt_input_field。错误：{e}")
                return

        client = ComfyUIClient(comfyui_base_url)
        poll_interval = float(self.config.get("poll_interval_sec", 1.0))
        job_timeout = int(self.config.get("job_timeout_sec", 180))
        image_index = int(self.config.get("output_image_index", 0))
        if output_image_index_override is not None:
            image_index = int(output_image_index_override)

        try:
            prompt_id = await client.queue_prompt(workflow)
            img_ref = await self._wait_for_output_image(
                client=client,
                prompt_id=prompt_id,
                output_node_id=output_node_id,
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
