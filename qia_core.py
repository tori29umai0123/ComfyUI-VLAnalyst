from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = {"qwen3_5", "glm4v"}
OFFICIAL_MODEL_REPOS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "zai-org/GLM-4.6V-Flash",
)


def quantization_method(config: dict[str, Any]) -> str | None:
    quant = config.get("quantization_config")
    if not quant:
        return None
    if not isinstance(quant, dict):
        return "unknown"
    return str(quant.get("quant_method", "unknown")).lower()


def read_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise ValueError(f"config.json がありません: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"config.json を読めません: {path}: {exc}") from exc


def validate_model(model_path: str | Path) -> dict[str, Any]:
    config = read_config(model_path)
    model_type = str(config.get("model_type", ""))
    architectures = [str(x) for x in config.get("architectures", [])]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"未対応モデル形式です (model_type={model_type or '不明'})。"
            "Qwen3.5またはGLM-4.6V-FlashのTransformers形式だけを対象にします。"
        )
    if not any("ForConditionalGeneration" in x for x in architectures):
        raise ValueError(f"画像対応のConditionalGenerationモデルではありません: {architectures}")
    quant = config.get("quantization_config")
    if quant:
        method = quant.get("quant_method", "unknown") if isinstance(quant, dict) else "unknown"
        raise ValueError(
            f"量子化形式 {method} は対応していません。非量子化モデルを使用してください。"
        )
    return config


def discover_models(roots: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for root_text in roots:
        root = Path(root_text)
        if not root.is_dir():
            continue
        candidates = [root] + [p for p in root.rglob("config.json")]
        for item in candidates:
            directory = item if item.is_dir() else item.parent
            try:
                cfg = read_config(directory)
            except ValueError:
                continue
            if cfg.get("model_type") not in SUPPORTED_MODEL_TYPES:
                continue
            try:
                label = str(directory.relative_to(root))
            except ValueError:
                label = directory.name
            if label == ".":
                label = directory.name
            key = label
            suffix = 2
            while key in found and found[key] != str(directory.resolve()):
                key = f"{label} ({suffix})"
                suffix += 1
            found[key] = str(directory.resolve())
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def download_official_model(repo_id: str, destination_root: str | Path) -> str:
    """Download one allow-listed official model and return its local directory."""
    if repo_id not in OFFICIAL_MODEL_REPOS:
        raise ValueError(f"自動ダウンロード対象外のモデルです: {repo_id}")
    destination = Path(destination_root) / repo_id.rsplit("/", 1)[-1]
    try:
        validate_model(destination)
        return str(destination.resolve())
    except ValueError:
        pass

    destination.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            # A single worker is slower but avoids connection resets seen with
            # large sharded checkpoints on some Windows/network combinations.
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(destination),
                max_workers=1,
                etag_timeout=30,
            )
            validate_model(destination)
            return str(destination.resolve())
        except Exception as exc:
            last_error = exc
            if os.name == "nt" and "10054" in str(exc):
                _configure_huggingface_ipv4()
            if attempt < 4:
                delay = 2 ** attempt
                print(f"[VL Analyst] download retry {attempt}/3 in {delay}s: {exc}")
                time.sleep(delay)
    raise RuntimeError(
        f"モデルのダウンロードに4回失敗しました: {repo_id} -> {destination}: {last_error}"
    ) from last_error


def _configure_huggingface_ipv4() -> None:
    """Retry Hugging Face over IPv4 when IPv6 TLS connections are reset."""
    import httpx
    from huggingface_hub import close_session, set_client_factory

    def client_factory():
        return httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            follow_redirects=True,
            timeout=None,
        )

    close_session()
    set_client_factory(client_factory)


def dtype_from_name(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


@dataclass
class CacheEntry:
    key: tuple[str, str, str]
    model: Any
    processor: Any
    config: dict[str, Any]
    position: str
    loaded_seconds: float
    in_use: int = 0


class ModelCache:
    """Owns at most one model. Handles passed through ComfyUI remain lightweight."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._entry: CacheEntry | None = None

    def _cleanup_cuda(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _drop_locked(self) -> str:
        if self._entry is None:
            return "モデルはロードされていません。"
        if self._entry.in_use:
            raise RuntimeError("モデルは現在推論中のため解放できません。")
        old = self._entry
        self._entry = None
        del old.model
        del old.processor
        del old
        gc.collect()
        self._cleanup_cuda()
        return "VL Analystのキャッシュからモデルを解放しました。"

    def release(self, expected_key: tuple[str, str, str] | None = None) -> str:
        with self._condition:
            if expected_key and self._entry and self._entry.key != expected_key:
                return "指定されたモデルは現在キャッシュされていません。"
            return self._drop_locked()

    def load(self, key: tuple[str, str, str], config: dict[str, Any]) -> tuple[bool, float]:
        with self._condition:
            if self._entry and self._entry.key == key:
                return True, 0.0
            if self._entry:
                self._drop_locked()

            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPUが認識されていません。この初期版はGPU推論専用です。")
            path, dtype_name, attention = key
            started = time.perf_counter()
            kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": dtype_from_name(dtype_name),
                "low_cpu_mem_usage": True,
            }
            if attention != "auto":
                kwargs["attn_implementation"] = attention
            try:
                processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
                model.eval()
                model.to("cuda")
            except Exception:
                for name in ("model", "processor"):
                    if name in locals():
                        del locals()[name]
                gc.collect()
                self._cleanup_cuda()
                raise
            elapsed = time.perf_counter() - started
            self._entry = CacheEntry(key, model, processor, config, "gpu", elapsed)
            return False, elapsed

    @contextmanager
    def use(self, key: tuple[str, str, str]):
        with self._condition:
            if self._entry is None or self._entry.key != key:
                raise RuntimeError("モデルキャッシュがありません。Loaderを再実行してください。")
            entry = self._entry
            while entry.in_use:
                self._condition.wait()
                if self._entry is not entry:
                    raise RuntimeError("待機中にモデルキャッシュが変更されました。Loaderを再実行してください。")
            if entry.position == "cpu":
                started = time.perf_counter()
                entry.model.to("cuda")
                entry.position = "gpu"
                restore_seconds = time.perf_counter() - started
            else:
                restore_seconds = 0.0
            entry.in_use = 1
        try:
            yield entry, restore_seconds
        finally:
            # finish() performs placement while the entry remains protected.
            pass

    def finish(self, key: tuple[str, str, str], mode: str) -> str:
        with self._condition:
            if not self._entry or self._entry.key != key:
                return "not_cached"
            if self._entry.in_use != 1:
                raise RuntimeError("内部エラー: 推論所有権が不正です。")
            self._entry.in_use = 0
            self._condition.notify_all()
            if mode == "gpu":
                return self._entry.position
            if mode == "cpu":
                try:
                    self._entry.model.to("cpu")
                except Exception as exc:
                    raise RuntimeError(f"このモデルはCPU退避できません: {exc}") from exc
                self._entry.position = "cpu"
                self._cleanup_cuda()
                return "cpu"
            self._drop_locked()
            return "unloaded"


MODEL_CACHE = ModelCache()


def tensor_to_pil(image_tensor):
    import numpy as np
    from PIL import Image

    array = image_tensor.detach().cpu().float().clamp(0, 1).numpy()
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"ComfyUI IMAGEのshapeが不正です: {array.shape}")
    array = (array * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, "RGB")


def resize_pil(image, max_long_edge: int):
    if max_long_edge <= 0 or max(image.size) <= max_long_edge:
        return image
    scale = max_long_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size)


def image_sha256(images: list[Any]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.tobytes())
        digest.update(str(image.size).encode("ascii"))
    return digest.hexdigest()


def build_prompt(processor, messages: list[dict[str, Any]], add_vision_id: bool = False) -> str:
    template_owner = processor if getattr(processor, "chat_template", None) else processor.tokenizer
    try:
        return template_owner.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            add_vision_id=add_vision_id,
        )
    except TypeError as exc:
        raise RuntimeError(
            "このprocessorのchat templateはenable_thinking=Falseを受け付けません。"
            "Thinking無効を保証できないため実行を中止しました。"
        ) from exc


def runtime_versions() -> dict[str, Any]:
    import torch
    import transformers
    import huggingface_hub

    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
    }


def run_inference(
    handle: dict[str, Any], images: list[Any], *, is_video: bool, timestamps: list[float] | None,
    sample_fps: float, system_prompt: str, user_prompt: str, max_new_tokens: int,
    do_sample: bool, temperature: float, top_p: float, top_k: int, seed: int,
    json_mode: bool, keep_mode: str, source_meta: dict[str, Any],
) -> tuple[str, str, str]:
    import torch

    key = tuple(handle["cache_key"])
    started = time.perf_counter()
    if not images:
        raise ValueError("入力フレームが空です。")
    original_sizes = [list(im.size) for im in images]
    visuals = ([{"type": "video", "video": images}] if is_video else
               [{"type": "image", "image": image} for image in images])
    if json_mode:
        user_prompt += "\nReturn exactly one valid JSON value. Do not use Markdown fences."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [*visuals, {"type": "text", "text": user_prompt}]},
    ]
    cuda_before: dict[str, int] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cuda_before = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    raw = ""
    json_valid: bool | None = None
    json_error: str | None = None
    output_tokens = 0
    preprocess_seconds = inference_seconds = restore_seconds = 0.0
    try:
        with MODEL_CACHE.use(key) as (entry, restore_seconds):
            prep = time.perf_counter()
            video_kwargs = {}
            if is_video:
                source_fps = float(source_meta.get("source_fps", sample_fps))
                frame_indices = source_meta.get("frame_indices", list(range(len(images))))
                video_kwargs = {"do_sample_frames": False, "video_metadata": [{
                    "total_num_frames": int(source_meta.get("source_frame_count", len(images))),
                    "fps": source_fps,
                    "duration": source_meta.get("duration_seconds"),
                    "width": images[0].width, "height": images[0].height,
                    "frames_indices": frame_indices,
                }]}
            if handle["model_type"] == "glm4v":
                inputs = entry.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt", processor_kwargs=video_kwargs,
                )
                inputs.pop("token_type_ids", None)
            else:
                prompt = build_prompt(entry.processor, messages, add_vision_id=not is_video and len(images) > 1)
                kwargs = {"text": [prompt], "return_tensors": "pt", **video_kwargs}
                if is_video:
                    kwargs["videos"] = [images]
                else:
                    kwargs["images"] = images
                inputs = entry.processor(**kwargs)
            inputs = {name: value.to("cuda") if hasattr(value, "to") else value for name, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[-1])
            preprocess_seconds = time.perf_counter() - prep
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            generation: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample),
            }
            if do_sample:
                generation.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
            infer = time.perf_counter()
            with torch.inference_mode():
                generated = entry.model.generate(**inputs, **generation)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - infer
            new_tokens = generated[:, input_tokens:]
            output_tokens = int(new_tokens.shape[-1])
            raw = entry.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            del generated, new_tokens, inputs
        if json_mode:
            try:
                json.loads(raw)
                json_valid = True
            except json.JSONDecodeError as exc:
                json_valid, json_error = False, str(exc)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA OOM: model={handle['model_path']}, frames={len(images)}, "
            f"sizes={original_sizes}, max_new_tokens={max_new_tokens}. 設定は自動変更していません。"
        ) from exc
    finally:
        placement = MODEL_CACHE.finish(key, keep_mode)
        gc.collect()

    cuda_peak: dict[str, int] = {}
    if torch.cuda.is_available():
        cuda_peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
    total = time.perf_counter() - started
    metadata = {
        "run_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": handle["model_label"], "model_path": handle["model_path"],
        "model_type": handle["model_type"], "backend": "transformers", "dtype": handle["dtype"],
        "quantization": handle.get("quantization"),
        "thinking": False, "cache_hit_at_loader": handle["cache_hit"], "load_seconds": handle["load_seconds"],
        "gpu_restore_seconds": restore_seconds, "final_placement": placement,
        "input_kind": "video_frames" if is_video else "images" if len(images) > 1 else "image",
        "frame_count": len(images),
        "timestamps_seconds": timestamps, "original_sizes": original_sizes,
        "image_sha256": image_sha256(images), **source_meta,
        "system_prompt": system_prompt, "user_prompt": user_prompt,
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                       "temperature": temperature if do_sample else None, "top_p": top_p if do_sample else None,
                       "top_k": top_k if do_sample else None, "seed": seed},
        "output": raw, "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / inference_seconds if inference_seconds else None,
        "json_requested": json_mode, "json_valid": json_valid, "json_error": json_error,
        "timing_seconds": {"preprocess": preprocess_seconds, "inference": inference_seconds, "total": total},
        "cuda_memory_bytes": {"start": cuda_before, "process_peak": cuda_peak,
                              "note": "PyTorch process-wide statistics; not model-only VRAM."},
        "versions": runtime_versions(),
    }
    token = json.dumps({"cache_key": list(key), "run_id": metadata["run_id"]}, ensure_ascii=False)
    return raw, json.dumps(metadata, ensure_ascii=False, indent=2), token
