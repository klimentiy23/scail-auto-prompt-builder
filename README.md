# SCAIL Auto Prompt Builder for ComfyUI

A small ComfyUI custom node that builds a ready-to-use prompt for Wan2.1 / SCAIL-2 character or object replacement workflows from:

- a reference image; and
- sampled source video frames.

It is designed for workflows where the user wants to drop in a reference photo and a video, then press **Queue** without manually writing a long replacement prompt.

## Node

Display names:

```text
SCAIL Auto Prompt Builder
SCAIL Auto Prompt Builder V2 (Target + MultiRef JSON)
SCAIL Full-Length / Preview Planner
```

Class names:

```text
SCAILAutoPromptBuilder
SCAILAutoPromptBuilderV2
SCAILFullLengthPlanner
```

Categories:

```text
SCAIL/Prompt
SCAIL/Utilities
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

## V2: target selection, multi-reference identity, structured JSON

`SCAIL Auto Prompt Builder V2 (Target + MultiRef JSON)` adds the controls needed for more predictable replacement workflows:

- **Target selection** — describe who/what to replace, for example `main woman`, `woman on the left`, `person #2 in red shirt`, or `foreground dog`.
- **Multi-reference identity** — ordered inputs: 1) face close-up, 2) body 3/4 view, 3) body front view, 4) body back view, plus optional extra reference.
- **Structured task JSON** — the VLM is asked to return explicit fields: `task_mode`, `target_subject`, `reference_identity`, `replace`, `preserve`, `avoid`, `positive_prompt`, `negative_prompt`, and `confidence_notes`.
- **Positive + negative prompts** — V2 outputs the final positive prompt and a generated negative prompt separately.
- **Render plan output** — V2 reports input frames, expected output frames, internal 4n+1 frame padding, SCAIL chunk lengths, and a preview/full-render note.

The V2 node still uses local models only and keeps `unload_after=True` by default so VRAM is released before Wan/SCAIL video generation.

## Full-length / preview planner

`SCAIL Full-Length / Preview Planner` is a lightweight utility node for long videos. It accepts the video frame batch and reports:

```text
input: 272 frames
output expected: 272 frames
internal planned frames: 273
chunks: 4 [81, 81, 81, 45]
frame_load_cap to use: 0 (0/unlimited full render)
rough estimate: ~98.0 minutes
```

Use `mode=preview_81` to plan a first 81-frame test, then switch to full length by setting the video loader frame cap to `0` / unlimited for the final render.

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

V2 adds:

| Input | Type | Default | Notes |
|---|---|---:|---|
| `task_mode` | COMBO | `character_replacement` | Character, face identity, outfit, or object replacement. Background replacement is intentionally out of scope. |
| `target_selection` | STRING | `main foreground subject` | Human-readable target selector: `woman on the left`, `person #2`, `main dancer`, etc. |
| `face_reference_image` | IMAGE | required | Face close-up / identity reference. |
| `body_3_4_reference_image` | IMAGE | optional | 3/4 body view: volume, silhouette, side/front mix. |
| `body_front_reference_image` | IMAGE | optional | Full-body front view: proportions and outfit/front details. |
| `body_back_reference_image` | IMAGE | optional | Full-body back view: rear silhouette, hair/back/outfit details. |
| `extra_reference_image` | IMAGE | optional | Any additional identity evidence. |

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

## Example workflow / blueprint

This repository includes an experimental V2 example workflow and blueprint under:

```text
examples/workflows/video_wan21_scail2_character_replacement_16gb_auto_extend_v2_target_multiref_preview.json
examples/workflows/Character Replacement (SCAIL-2 16GB Auto Extend V2 Target MultiRef Preview).json
```

The V2 workflow keeps the original working one-button workflow separate and adds target selection, ordered identity references (1 face close-up, 2 body 3/4, 3 body front, 4 body back), structured JSON preview, generated negative prompt, and preview/full-length planning. It still requires the companion SCAIL Auto Extend node for actual long-video chunking/stitching.

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
