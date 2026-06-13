# Cosmos 3 Showcase — Reason → Generate

This showcase demonstrates the **full Cosmos 3 omnimodal loop** in strands-cosmos,
running entirely on a single local NVIDIA L40S (no NIM):

1. **Reason** about a real input video with `Cosmos3ReasonerModel` (vLLM)
2. **Generate** similar videos — including one **with synchronized audio** —
   from the reasoner's own description, using `Cosmos3GeneratorModel` (Diffusers)

All artifacts in this folder are produced by the code in this PR.

---

## Input video

The original dashcam-style clip of a construction site (720×1280, 186 frames):

<img src="00_input.gif" width="180" alt="Input video"/>

---

## Step 1 — Reasoning (`Cosmos3ReasonerModel`, vLLM)

### Detailed caption
> Two construction workers wearing yellow safety vests and helmets are walking away from the camera on a dirt path within a bustling construction site. The ground is covered in loose soil, with visible tire tracks crisscrossing the surface. In the background, a large yellow front-end loader moves slowly across the site, its bucket raised slightly as it navigates the terrain. Behind the loader, partially obscured by rebar and concrete slabs, an excavator operates near a foundation area. The scene is framed by urban buildings in the distance, including a distinctive church-like structure with a tall spire and modern glass-fronted buildings. The overall atmosphere suggests active progress on a significant infrastructure project under clear daylight conditions.

### Temporal localization
> From 0.3 to 4.8 seconds, a yellow front loader is driving on a dirt path in the foreground of a construction site, moving away from the camera. Two individuals wearing bright green safety vests and yellow hard hats are walking towards the back, engaged in conversation. The woman occasionally gestures with her hand as they walk. In the background, an excavator operates near rebar structures, while parked cars line the street behind the site. Buildings and trees frame the urban setting under a clear sky.

### Embodied next-action (with `<think>` reasoning)
> <think>
Okay, let's see. The video is a construction site with two workers in safety gear walking through the dirt. The camera is static, showing a wide shot. The workers are on a dirt path, moving forward, with the woman gesturing and talking. There's a backhoe loader and some construction materials around.

So, what's next? Well, since they're on a construction site, their actions would be related to their work. They might be inspecting the site, discussing plans, or moving towards a specific area. The presence of the backhoe suggests they might be near excavation or earthmoving. The woman gesturing could mean she's directing the man's attention to something, maybe pointing out an area that needs work. 

Since they're both walking and she's gesturing, maybe they're heading towards the backhoe or another part of the site. The backhoe is stationary, so maybe they need to use it next. Alternatively, they could be checking progress,

### Distilled generation prompt
The reasoner condensed its own understanding into a prompt for the generator:
> **Two construction workers in yellow safety vests and helmets walk across a dusty site, gesturing toward a yellow front loader and distant excavator as they converse.**

---

## Step 2 — Generation (`Cosmos3GeneratorModel`, Diffusers, 480p)

All three generated from the reasoner-derived prompt above (832×480, 49 frames @ 16fps, 30 steps).

### text→video
<img src="01_text2video.gif" width="416" alt="text2video"/>

`cosmos3_text2video(prompt=...)` → [01_text2video.mp4](01_text2video.mp4)

### text→video **with sound** (H264 + AAC stereo 48kHz)
<img src="02_text2video_sound.gif" width="416" alt="text2video with sound"/>

`cosmos3_text2video_sound(prompt=...)` → [02_text2video_sound.mp4](02_text2video_sound.mp4)
*(GIF has no audio; the MP4 carries a stereo AAC track — `ffprobe` confirms `codec_type=audio, channels=2, sample_rate=48000`.)*

### image→video (continuing from a frame of the original)
<img src="03_image2video.gif" width="416" alt="image2video"/>

`cosmos3_image2video(prompt=..., image=frame.jpg)` → [03_image2video.mp4](03_image2video.mp4)

---

## How it was produced

```python
from strands_cosmos import Cosmos3ReasonerModel, Cosmos3GeneratorModel

# 1) Reason (vLLM server running Cosmos3-Nano)
reasoner = Cosmos3ReasonerModel(base_url="http://localhost:8000/v1")
caption = reasoner("Caption this video in detail: <video>input.mp4</video>")

# 2) Generate from that understanding (Diffusers, in-process)
gen = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")
gen.generate(mode="text2video",            prompt=prompt, out_path="01_text2video.mp4", resolution="480")
gen.generate(mode="text2video-with-sound", prompt=prompt, out_path="02_text2video_sound.mp4", enable_sound=True)
gen.generate(mode="image2video",           prompt=prompt, image="frame.jpg", out_path="03_image2video.mp4")
```

> **Single-GPU note:** the reasoner and generator each load a 16B model, so on one
> ~46GB GPU we stop the vLLM reasoner before running the Diffusers generator.

**Timings (L40S, 480p / 49f / 30 steps):** text2video 55.5s · text2video+sound 43.2s · image2video 42.1s.
Reasoning: caption 5.2s · temporal 2.8s · embodied 5.0s.
