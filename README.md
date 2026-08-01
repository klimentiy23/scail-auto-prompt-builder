# SCAIL Auto Prompt Builder for ComfyUI

A small ComfyUI custom node that builds a ready-to-use prompt for Wan2.1 / SCAIL-2 character or object replacement workflows from:

- a reference image; and
- sampled source video frames.

It is designed for workflows where the user wants to drop in a reference photo and a video, then press **Queue** without manually writing a long replacement prompt.

## Node

Display name:

```text
SCAIL Auto Prompt Builder
```

Class name:

```text
SCAILAutoPromptBuilder
```

Category:

```text
SCAIL/Prompt
```

## What it does

The node samples:

- the reference image;
- first / middle / last frames from the source video frame batch;

then asks a local vision-language model to write a SCAIL-friendly positive prompt that:

- replaces only the main target subject;
- uses the reference image only for identity/appearance;
- preserves source video clothing, background, pose, lighting, camera movement, timing, and non-target people/objects;
- adds continuity/artifact prevention language.

The output is a plain `STRING`, intended to feed directly into `CLIPTextEncode` / positive prompt conditioning.

A second `diagnostics` string output reports which model/device was used and whether VLM generation succeeded.

## Local model layout

By default, the node looks for local VLMs under:

```text
D:/comfyui/models/LLM
```

It was tested with:

```text
D:/comfyui/models/LLM/Qwen-VL/Qwen3-VL-2B-Instruct
```

No downloads are performed by this node. The model must already exist locally.

You can override the root with:

```bash
COMFYUI_MODELS_DIR=D:/comfyui/models
```

## Inputs

| Input | Type | Default | Notes |
|---|---|---:|---|
| `reference_image` | IMAGE | required | Subject identity / appearance source. |
| `video_frames` | IMAGE | required | Source video frames, usually from `LoadVideo` / resized pose video. |
| `model_folder` | COMBO | auto-detected | Relative to `D:/comfyui/models/LLM`. |
| `device` | `cpu` / `cuda` | `cpu` in node schema | `cuda` is faster, but use `unload_after=True`. |
| `max_side` | INT | 768 | Resize images before VLM. |
| `max_new_tokens` | INT | 700 | Prompt generation length cap. |
| `temperature` | FLOAT | 0.2 | Lower = more deterministic. |
| `unload_after` | BOOLEAN | true | Clears cached VLM after prompt generation to free VRAM/RAM for video generation. |
| `fail_mode` | COMBO | `fallback_template` | On VLM failure, either emit a generic safe SCAIL prompt or raise. |
| `user_hint` | STRING | empty | Optional target instruction, e.g. `replace only the woman in black cowboy outfit`. |

## Recommended workflow wiring

```text
LoadImage reference
LoadVideo source
        ↓
SCAIL Auto Prompt Builder
        ↓
Show Text / preview prompt
        ↓
CLIPTextEncode positive
        ↓
SCAIL Auto Extend V3
        ↓
SaveVideo
```

For one-button use, leave `user_hint` empty. If needed, type a short instruction such as:

```text
replace the main woman only, keep the hat and handbag from the video
```

## Performance notes

On the original test machine with an RTX 5060 Ti 16 GB:

- CPU first generation with Qwen3-VL-2B took about 1–2 minutes.
- CUDA generation took about 16 seconds in a dummy smoke test.

For full SCAIL workflows, keep:

```text
unload_after=True
```

so the VLM is released before Wan/SCAIL video generation starts.

## Dependencies

The node relies on packages normally present in a modern ComfyUI portable environment:

- `torch`
- `transformers`
- `Pillow`
- `numpy`

It uses `AutoProcessor` and `AutoModelForImageTextToText` with `local_files_only=True`.

## Safety / limitations

- This node does not download models.
- This node does not send images to external APIs.
- It only generates text prompts; it does not perform masking, detection, or face recognition.
- The generated prompt should be previewed before long renders, especially for ambiguous videos.

## License

MIT
