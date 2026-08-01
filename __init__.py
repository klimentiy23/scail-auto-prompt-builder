import os
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None


_MODEL_CACHE = {}


def _models_root():
    # User's external ComfyUI model root. Keep this on D:, no downloads.
    env = os.environ.get("COMFYUI_MODELS_DIR")
    if env and Path(env).exists():
        return Path(env)
    return Path("D:/comfyui/models")


def _find_vlm_models():
    root = _models_root()
    found = []
    llm = root / "LLM"
    if llm.exists():
        for cfg in llm.rglob("config.json"):
            try:
                txt = cfg.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                txt = ""
            rel = str(cfg.parent.relative_to(llm)).replace("\\", "/")
            if any(k in txt or k in rel.lower() for k in ["vl", "vision", "qwen3_vl", "qwen2_vl", "llava", "minicpm", "internvl"]):
                found.append(rel)
    # Stable preferred order for this machine.
    found = sorted(set(found), key=lambda x: ("qwen3-vl-2b" not in x.lower(), x.lower()))
    return found or ["Qwen-VL/Qwen3-VL-2B-Instruct"]


def _tensor_to_pil(img):
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr).convert("RGB")


def _resize_for_vlm(im, max_side):
    if max_side <= 0:
        return im
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def _sample_images(batch, max_frames=3, max_side=768):
    if batch is None:
        return []
    try:
        n = int(batch.shape[0])
    except Exception:
        n = len(batch)
    if n <= 0:
        return []
    if max_frames <= 1 or n == 1:
        idxs = [0]
    elif n == 2:
        idxs = [0, 1]
    else:
        idxs = sorted(set([0, n // 2, n - 1]))[:max_frames]
    return [_resize_for_vlm(_tensor_to_pil(batch[i]), max_side) for i in idxs]


def _fallback_prompt(user_hint="", reference_caption="", video_caption=""):
    hint = (user_hint or "").strip()
    ref = (reference_caption or "").strip()
    vid = (video_caption or "").strip()
    parts = [
        "Replace only the main foreground person/character/object in the source video with the subject from the reference image.",
        "Use the reference image only for identity and physical appearance: face, hair, skin tone, body type, and recognizable subject features.",
        "Preserve the source video's original pose, motion, timing, camera movement, framing, background, lighting, shadows, color grading, clothing, accessories, and interactions unless the user explicitly says otherwise.",
        "Do not copy the reference image background, lighting, pose, clothing, or composition into the video.",
    ]
    if ref:
        parts.append("Reference subject description: " + ref)
    if vid:
        parts.append("Source video description: " + vid)
    if hint:
        parts.append("User instruction: " + hint)
    parts.append(
        "Maintain one consistent identity across all frames. No identity drift, no duplicate subject, no extra faces, no face applied to background people, no extra limbs, no deformed hands, no flicker, no ghosting, no object duplication, and no changing clothes or accessories."
    )
    return "\n\n".join(parts)


class SCAILAutoPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "model_folder": (_find_vlm_models(), {"default": _find_vlm_models()[0]}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "max_side": ("INT", {"default": 768, "min": 256, "max": 1536, "step": 64}),
                "max_new_tokens": ("INT", {"default": 700, "min": 128, "max": 1600, "step": 32}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.2, "step": 0.05}),
                "unload_after": ("BOOLEAN", {"default": True}),
                "fail_mode": (["fallback_template", "raise_error"], {"default": "fallback_template"}),
                "user_hint": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "diagnostics")
    FUNCTION = "build_prompt"
    CATEGORY = "SCAIL/Prompt"
    DESCRIPTION = "Builds a SCAIL-2 character/object replacement prompt from a reference image and sampled video frames using a local VLM under D:/comfyui/models/LLM; no downloads."

    def _load_qwen_vl(self, model_folder, device):
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        root = _models_root() / "LLM" / model_folder
        if not root.exists():
            raise FileNotFoundError(f"Local VLM folder not found: {root}")
        key = (str(root), device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
        kwargs = {"local_files_only": True, "trust_remote_code": True, "low_cpu_mem_usage": True}
        if device == "cuda" and torch.cuda.is_available():
            kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "cuda"})
        else:
            kwargs.update({"torch_dtype": torch.float32, "device_map": "cpu"})
        model = AutoModelForImageTextToText.from_pretrained(str(root), **kwargs)
        model.eval()
        _MODEL_CACHE[key] = (processor, model)
        return processor, model

    def _vlm_prompt(self, processor, model, images, user_hint, max_new_tokens, temperature):
        instruction = f"""
You are writing a single clean English prompt for a ComfyUI Wan2.1 SCAIL-2 character/object replacement workflow.
The first image is the reference subject. The following images are sampled frames from the source video.
Task: replace only the main foreground subject in the source video with the reference subject.

Write the final diffusion prompt only, no headings and no markdown.
Requirements:
- Identify the reference subject's stable identity traits: face, hair, skin tone, body type, age category, distinctive visible features.
- Identify the source video's subject, clothing, accessories, pose/motion, setting, lighting, camera/framing.
- Preserve the source video clothing, accessories, background, lighting, camera motion, pose timing, interactions and all non-target people/objects.
- Use the reference image only for identity/appearance. Do not copy its background, pose, lighting, clothes, or composition.
- Add strong continuity and artifact prevention: no identity drift, no duplicated face/body, no extra limbs/fingers, no flicker, no ghosting, no changing clothes/accessories, no face transfer to background people.
- If the user hint conflicts with preservation, follow the user hint only for the intended replacement target.
User hint: {user_hint or 'none'}
""".strip()
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        do_sample = temperature > 0.01
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=max(temperature, 0.01) if do_sample else None)
        # strip prompt tokens when possible
        if "input_ids" in inputs:
            generated = generated[:, inputs["input_ids"].shape[-1]:]
        out = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return out

    def build_prompt(self, reference_image, video_frames, model_folder, device, max_side, max_new_tokens, temperature, unload_after, fail_mode, user_hint=""):
        diagnostics = []
        ref_imgs = _sample_images(reference_image, 1, max_side)
        vid_imgs = _sample_images(video_frames, 3, max_side)
        images = ref_imgs + vid_imgs
        diagnostics.append(f"images_used={len(images)} reference={len(ref_imgs)} video_samples={len(vid_imgs)} model={model_folder} device={device}")
        try:
            processor, model = self._load_qwen_vl(model_folder, device)
            prompt = self._vlm_prompt(processor, model, images, user_hint, max_new_tokens, temperature)
            if not prompt or len(prompt) < 80:
                raise RuntimeError(f"VLM returned too little text: {prompt!r}")
            diagnostics.append("vlm=ok")
        except Exception as e:
            diagnostics.append(f"vlm=failed {type(e).__name__}: {e}")
            if fail_mode == "raise_error":
                raise
            prompt = _fallback_prompt(user_hint=user_hint)
        finally:
            if unload_after:
                # Drop all cached models to leave VRAM/RAM for Wan/SCAIL generation.
                _MODEL_CACHE.clear()
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # enforce single clean text block
        prompt = "\n".join(line.rstrip() for line in prompt.replace("```", "").splitlines()).strip()
        return (prompt, "\n".join(diagnostics))


def _extract_json_object(text):
    text = (text or "").strip().replace("```json", "```")
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except Exception:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip(" -\t") for x in value.splitlines() if x.strip(" -\t")]
    return [str(value)]


def _structured_fallback(task_mode, target_selection, user_hint):
    target = (target_selection or "main foreground subject").strip() or "main foreground subject"
    hint = (user_hint or "").strip()
    return {
        "task_mode": task_mode,
        "target_subject": target,
        "reference_identity": "the subject in the reference image, using identity, face, hair, skin tone, body type, and stable recognizable features",
        "replace": ["identity", "face", "hair", "body appearance", "recognizable subject features"],
        "preserve": ["source video pose", "motion timing", "camera movement", "framing", "background", "lighting", "shadows", "clothing", "accessories", "interactions", "all non-target people and objects"],
        "avoid": ["identity drift", "duplicated face or body", "extra limbs or fingers", "flicker", "ghosting", "changing clothes or accessories", "applying the face to background people", "copying the reference background"],
        "user_hint": hint,
    }


def _structured_to_prompt(data):
    if isinstance(data, dict) and data.get("positive_prompt"):
        return str(data["positive_prompt"]).strip()
    target = str(data.get("target_subject") or "main foreground subject").strip()
    ref = str(data.get("reference_identity") or "the subject from the reference images").strip()
    replace = ", ".join(_as_list(data.get("replace")))
    preserve = ", ".join(_as_list(data.get("preserve")))
    avoid = ", ".join(_as_list(data.get("avoid")))
    hint = str(data.get("user_hint") or "").strip()
    parts = [
        f"Replace only {target} in the source video with {ref}.",
    ]
    if replace:
        parts.append(f"Replace: {replace}.")
    if preserve:
        parts.append(f"Preserve exactly: {preserve}.")
    parts.append("Use the reference images only for identity and stable appearance, not for background, pose, lighting, clothing, or composition unless explicitly requested.")
    if hint:
        parts.append(f"User instruction: {hint}.")
    if avoid:
        parts.append(f"Avoid: {avoid}.")
    parts.append("Maintain one consistent identity across all frames with temporally stable details and natural motion continuity.")
    return "\n\n".join(parts)


def _negative_from_structured(data):
    if isinstance(data, dict) and data.get("negative_prompt"):
        return str(data["negative_prompt"]).strip()
    avoid = _as_list(data.get("avoid") if isinstance(data, dict) else None)
    base = [
        "identity drift", "wrong target subject", "background person changed", "duplicated face", "duplicated body",
        "extra limbs", "extra fingers", "deformed hands", "flicker", "ghosting", "warped face", "melted face",
        "changing clothing", "changing accessories", "copied reference background", "wrong lighting", "unstable mask edges",
    ]
    return ", ".join(dict.fromkeys(base + avoid))


def _plan_chunks(n_frames, chunk_len=81, overlap=5):
    import math
    if n_frames <= 0:
        return 0, []
    n_eff = math.ceil((n_frames - 1) / 4) * 4 + 1
    chunk_len = ((chunk_len - 1) // 4) * 4 + 1
    if overlap % 4 != 1:
        overlap = max(1, ((overlap - 1) // 4) * 4 + 1)
    if n_eff <= chunk_len:
        return n_eff, [n_eff]
    step = chunk_len - overlap
    chunks = []
    remaining_start = 0
    while remaining_start + chunk_len < n_eff:
        chunks.append(chunk_len)
        remaining_start += step
    chunks.append(n_eff - remaining_start)
    return n_eff, chunks


class SCAILAutoPromptBuilderV2(SCAILAutoPromptBuilder):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference_image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "task_mode": (["character_replacement", "face_identity_replacement", "outfit_replacement", "object_replacement"], {"default": "character_replacement"}),
                "target_selection": ("STRING", {"default": "main foreground subject", "multiline": True}),
                "user_hint": ("STRING", {"default": "", "multiline": True}),
                "model_folder": (_find_vlm_models(), {"default": _find_vlm_models()[0]}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "max_side": ("INT", {"default": 768, "min": 256, "max": 1536, "step": 64}),
                "max_new_tokens": ("INT", {"default": 900, "min": 128, "max": 2000, "step": 32}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.2, "step": 0.05}),
                "unload_after": ("BOOLEAN", {"default": True}),
                "fail_mode": (["fallback_template", "raise_error"], {"default": "fallback_template"}),
            },
            "optional": {
                "body_3_4_reference_image": ("IMAGE",),
                "body_front_reference_image": ("IMAGE",),
                "body_back_reference_image": ("IMAGE",),
                "extra_reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "diagnostics", "task_json", "negative_prompt", "input_frames", "planned_output_frames", "render_plan")
    FUNCTION = "build_prompt_v2"
    CATEGORY = "SCAIL/Prompt"
    DESCRIPTION = "V2 SCAIL prompt builder: target selection text, multi-reference identity inputs, structured JSON plan, and preview/full-length render plan text."

    def _vlm_structured(self, processor, model, images, task_mode, target_selection, user_hint, max_new_tokens, temperature):
        instruction = f"""
You are controlling a ComfyUI Wan2.1 SCAIL-2 replacement workflow.
Images are in this order: face close-up reference, body 3/4 reference, body front reference, body back reference, optional extra reference if present, then sampled frames from the source video.

Return STRICT JSON only. No markdown. No prose outside JSON.

Schema:
{{
  "task_mode": "{task_mode}",
  "target_subject": "who/what in the source video should be replaced",
  "reference_identity": "stable identity and appearance from the reference images",
  "replace": ["identity/appearance fields to replace"],
  "preserve": ["video attributes that must stay unchanged"],
  "avoid": ["artifacts and wrong edits to prevent"],
  "positive_prompt": "final clean English diffusion prompt for SCAIL-2",
  "negative_prompt": "comma-separated negative prompt",
  "confidence_notes": "short note about ambiguity, if any"
}}

Rules:
- Target selection from user: {target_selection or 'main foreground subject'}
- User hint: {user_hint or 'none'}
- Preserve source video motion timing, pose, camera/framing, background, lighting, shadows, clothing/accessories, interactions, and non-target people/objects unless user explicitly says otherwise.
- Use reference images only for identity and stable appearance. Combine them by role: face close-up for facial identity, body 3/4 for volume/silhouette, body front for proportions/outfit, and body back for rear silhouette/hair/back details.
- Do not choose a different person than the target selection.
- Include continuity and artifact prevention language.
""".strip()
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        do_sample = temperature > 0.01
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                       temperature=max(temperature, 0.01) if do_sample else None)
        if "input_ids" in inputs:
            generated = generated[:, inputs["input_ids"].shape[-1]:]
        text_out = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text_out

    def build_prompt_v2(self, face_reference_image, video_frames, task_mode, target_selection, user_hint, model_folder, device,
                        max_side, max_new_tokens, temperature, unload_after, fail_mode,
                        body_3_4_reference_image=None, body_front_reference_image=None, body_back_reference_image=None, extra_reference_image=None):
        diagnostics = []
        ref_imgs = _sample_images(face_reference_image, 1, max_side)
        for optional in [body_3_4_reference_image, body_front_reference_image, body_back_reference_image, extra_reference_image]:
            ref_imgs.extend(_sample_images(optional, 1, max_side))
        vid_imgs = _sample_images(video_frames, 3, max_side)
        images = ref_imgs + vid_imgs
        try:
            input_frames = int(video_frames.shape[0])
        except Exception:
            input_frames = len(video_frames) if video_frames is not None else 0
        planned_frames, chunks = _plan_chunks(input_frames, 81, 5)
        render_plan = (f"input: {input_frames} frames\n"
                       f"output expected: {input_frames} frames\n"
                       f"internal planned frames: {planned_frames}\n"
                       f"chunks: {len(chunks)} {chunks}\n"
                       f"preview mode: set video loader frame cap to 81 for a first-chunk preview; full render uses frame cap 0/unlimited.")
        diagnostics.append(
            f"v2 images_used={len(images)} references={len(ref_imgs)} video_samples={len(vid_imgs)} "
            f"target={target_selection!r} task={task_mode} model={model_folder} device={device}"
        )
        data = None
        try:
            processor, model = self._load_qwen_vl(model_folder, device)
            raw = self._vlm_structured(processor, model, images, task_mode, target_selection, user_hint, max_new_tokens, temperature)
            data = _extract_json_object(raw)
            if not isinstance(data, dict):
                raise RuntimeError(f"VLM did not return parseable JSON: {raw[:300]!r}")
            data.setdefault("task_mode", task_mode)
            data.setdefault("target_subject", target_selection or "main foreground subject")
            data.setdefault("user_hint", user_hint or "")
            diagnostics.append("vlm_json=ok")
        except Exception as e:
            diagnostics.append(f"vlm_json=failed {type(e).__name__}: {e}")
            if fail_mode == "raise_error":
                raise
            data = _structured_fallback(task_mode, target_selection, user_hint)
        finally:
            if unload_after:
                _MODEL_CACHE.clear()
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        prompt = _structured_to_prompt(data).replace("```", "").strip()
        negative = _negative_from_structured(data).replace("```", "").strip()
        task_json = json.dumps(data, ensure_ascii=False, indent=2)
        return (prompt, "\n".join(diagnostics), task_json, negative, input_frames, input_frames, render_plan)


class SCAILFullLengthPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "mode": (["preview_81", "full_length"], {"default": "preview_81"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "chunk_length": ("INT", {"default": 81, "min": 9, "max": 1024, "step": 4}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 81, "step": 4}),
                "minutes_per_81_frame_chunk": ("FLOAT", {"default": 24.5, "min": 0.1, "max": 240.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("report", "frame_load_cap", "input_frames", "expected_output_frames", "estimated_minutes")
    FUNCTION = "plan"
    CATEGORY = "SCAIL/Utilities"
    DESCRIPTION = "Plans preview vs full-length SCAIL renders: frame count, chunk count, output length, and rough time estimate."

    def plan(self, video_frames, mode, fps, chunk_length, overlap, minutes_per_81_frame_chunk):
        try:
            input_frames = int(video_frames.shape[0])
        except Exception:
            input_frames = len(video_frames) if video_frames is not None else 0
        render_input = min(81, input_frames) if mode == "preview_81" else input_frames
        planned_frames, chunks = _plan_chunks(render_input, chunk_length, overlap)
        frame_load_cap = 81 if mode == "preview_81" else 0
        scale = max(1.0, chunk_length / 81.0)
        estimated = len(chunks) * minutes_per_81_frame_chunk * scale
        seconds = render_input / fps if fps else 0
        report = (f"mode: {mode}\n"
                  f"input: {input_frames} frames ({input_frames / fps:.2f}s at {fps:.2f} fps)\n"
                  f"render input: {render_input} frames ({seconds:.2f}s)\n"
                  f"output expected: {render_input} frames\n"
                  f"internal planned frames: {planned_frames}\n"
                  f"chunks: {len(chunks)} {chunks}\n"
                  f"frame_load_cap to use: {frame_load_cap} ({'81-frame preview' if frame_load_cap else '0/unlimited full render'})\n"
                  f"rough estimate: ~{estimated:.1f} minutes")
        return (report, frame_load_cap, input_frames, render_input, float(estimated))


NODE_CLASS_MAPPINGS = {
    "SCAILAutoPromptBuilder": SCAILAutoPromptBuilder,
    "SCAILAutoPromptBuilderV2": SCAILAutoPromptBuilderV2,
    "SCAILFullLengthPlanner": SCAILFullLengthPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILAutoPromptBuilder": "SCAIL Auto Prompt Builder",
    "SCAILAutoPromptBuilderV2": "SCAIL Auto Prompt Builder V2 (Target + MultiRef JSON)",
    "SCAILFullLengthPlanner": "SCAIL Full-Length / Preview Planner",
}
