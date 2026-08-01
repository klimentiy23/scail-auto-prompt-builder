import os
import gc
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


NODE_CLASS_MAPPINGS = {
    "SCAILAutoPromptBuilder": SCAILAutoPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILAutoPromptBuilder": "SCAIL Auto Prompt Builder",
}
