from __future__ import annotations

import json
import base64
import io
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps


st.set_page_config(
    page_title="Image to Physical Setup Translator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass
class Analysis:
    width: int
    height: int
    aspect_ratio: float
    brightness_mean: float
    brightness_p10: float
    brightness_p50: float
    brightness_p90: float
    highlight_ratio: float
    shadow_ratio: float
    contrast: float
    saturation: float
    warmth: float
    edge_density: float
    light_direction: str
    light_position: str
    light_x: float
    light_y: float
    light_sources: list[dict[str, Any]]
    shadow_x: float
    shadow_y: float
    depth_near_mean: float
    depth_near_ratio: float
    depth_complexity: float
    foreground_x: float
    foreground_y: float
    main_palette: list[str]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if high - low < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def image_to_array(image: Image.Image, size: int = 420) -> np.ndarray:
    copied = ImageOps.exif_transpose(image).convert("RGB")
    copied.thumbnail((size, size))
    return np.asarray(copied).astype(np.float32)


def weighted_centroid(values: np.ndarray, threshold: float) -> tuple[float, float]:
    mask = values >= threshold
    weights = np.where(mask, values, 0).astype(np.float32)
    if float(weights.sum()) <= 1e-6:
        h, w = values.shape
        return 0.5, 0.5

    h, w = values.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((xx * weights).sum() / weights.sum()) / max(w - 1, 1)
    cy = float((yy * weights).sum() / weights.sum()) / max(h - 1, 1)
    return round(cx, 3), round(cy, 3)


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    return components


def detect_light_sources(brightness_blur: np.ndarray, depth_map: np.ndarray) -> list[dict[str, Any]]:
    h, w = brightness_blur.shape
    threshold = max(float(np.percentile(brightness_blur, 86)), float(brightness_blur.mean() + brightness_blur.std() * 0.55))
    mask = brightness_blur >= threshold
    min_area = max(18, int(mask.size * 0.008))
    components = [component for component in connected_components(mask) if len(component) >= min_area]

    if not components:
        x, y = weighted_centroid(brightness_blur, float(np.percentile(brightness_blur, 88)))
        return [{
            "role": "key",
            "x": x,
            "y": y,
            "area_ratio": 0.0,
            "peak": round(float(brightness_blur.max()), 3),
            "mean": round(float(brightness_blur.mean()), 3),
            "depth": round(float(depth_map.mean()), 3),
        }]

    candidates: list[dict[str, Any]] = []
    for component in components:
        ys = np.array([p[0] for p in component])
        xs = np.array([p[1] for p in component])
        values = brightness_blur[ys, xs]
        weights = values + 0.001
        cx = float((xs * weights).sum() / weights.sum()) / max(w - 1, 1)
        cy = float((ys * weights).sum() / weights.sum()) / max(h - 1, 1)
        area_ratio = len(component) / mask.size
        candidates.append({
            "role": "fill",
            "x": round(cx, 3),
            "y": round(cy, 3),
            "area_ratio": round(float(area_ratio), 3),
            "peak": round(float(values.max()), 3),
            "mean": round(float(values.mean()), 3),
            "depth": round(float(depth_map[ys, xs].mean()), 3),
        })

    candidates.sort(key=lambda item: (item["peak"] * 0.55 + item["mean"] * 0.3 + item["area_ratio"] * 1.8), reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        too_close = any(abs(candidate["x"] - picked["x"]) < 0.16 and abs(candidate["y"] - picked["y"]) < 0.16 for picked in selected)
        if not too_close:
            selected.append(candidate)
        if len(selected) == 3:
            break

    for index, source in enumerate(selected):
        if index == 0:
            source["role"] = "key"
        elif source["x"] < 0.25 or source["x"] > 0.75 or source["y"] < 0.25:
            source["role"] = "rim"
        else:
            source["role"] = "fill"
    return selected


def infer_light_label(light_x: float, light_y: float) -> tuple[str, str]:
    horizontal = "left" if light_x < 0.42 else "right" if light_x > 0.58 else "center"
    vertical = "top" if light_y < 0.42 else "bottom" if light_y > 0.58 else "middle"

    if horizontal == "center" and vertical == "middle":
        return "front_diffused", "正面寄りの大きな拡散光"

    direction = f"{vertical}_{horizontal}"
    labels = {
        "top_left": "画面左上からの斜め光",
        "top_right": "画面右上からの斜め光",
        "middle_left": "画面左からのサイド光",
        "middle_right": "画面右からのサイド光",
        "bottom_left": "画面左下からの低い光",
        "bottom_right": "画面右下からの低い光",
        "top_center": "画面上からのトップ光",
        "bottom_center": "画面下からの低い正面光",
    }
    return direction, labels.get(direction, "方向性の弱い拡散光")


def get_palette(image: Image.Image, colors: int = 5) -> list[str]:
    small = ImageOps.exif_transpose(image).convert("RGB")
    small.thumbnail((240, 240))
    quantized = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = sorted(quantized.getcolors(), reverse=True)
    result: list[str] = []

    for _, index in counts[:colors]:
        base = index * 3
        rgb = tuple(palette[base : base + 3])
        if len(rgb) == 3:
            result.append(rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2]))))
    return result


def estimate_depth_map(arr: np.ndarray, gray: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    vertical_near = np.repeat(yy, w, axis=1)

    gx, gy = np.gradient(gray)
    gradient = normalize(np.sqrt(gx * gx + gy * gy))
    local_contrast = np.asarray(
        Image.fromarray((normalize(gray) * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    )
    local_contrast = normalize(local_contrast)

    sat_norm = normalize(saturation)
    darkness = 1 - normalize(gray)

    # This is a fast local proxy for relative depth. Higher values mean visually nearer.
    depth_near = (
        0.36 * vertical_near
        + 0.26 * local_contrast
        + 0.18 * gradient
        + 0.12 * sat_norm
        + 0.08 * darkness
    )
    blurred = Image.fromarray((normalize(depth_near) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
    return normalize(np.asarray(blurred, dtype=np.float32))


def analyze_image(image: Image.Image) -> tuple[Analysis, dict[str, Image.Image]]:
    arr = image_to_array(image)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    hsv = Image.fromarray(arr.astype(np.uint8)).convert("HSV")
    hsv_arr = np.asarray(hsv).astype(np.float32)
    saturation_map = hsv_arr[..., 1] / 255

    brightness = gray / 255
    brightness_blur = np.asarray(
        Image.fromarray((brightness * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(12)),
        dtype=np.float32,
    ) / 255
    depth_map = estimate_depth_map(arr, gray, saturation_map)

    edge_img = Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edge_img).astype(np.float32)

    light_x, light_y = weighted_centroid(brightness_blur, float(np.percentile(brightness_blur, 88)))
    shadow_x, shadow_y = weighted_centroid(1 - brightness_blur, float(np.percentile(1 - brightness_blur, 84)))
    foreground_x, foreground_y = weighted_centroid(depth_map, float(np.percentile(depth_map, 82)))
    light_sources = detect_light_sources(brightness_blur, depth_map)
    if light_sources:
        light_x = float(light_sources[0]["x"])
        light_y = float(light_sources[0]["y"])
    light_direction, light_position = infer_light_label(light_x, light_y)

    width, height = image.size
    analysis = Analysis(
        width=width,
        height=height,
        aspect_ratio=round(width / max(height, 1), 2),
        brightness_mean=round(float(brightness.mean()), 3),
        brightness_p10=round(float(np.percentile(brightness, 10)), 3),
        brightness_p50=round(float(np.percentile(brightness, 50)), 3),
        brightness_p90=round(float(np.percentile(brightness, 90)), 3),
        highlight_ratio=round(float((brightness > 0.78).mean()), 3),
        shadow_ratio=round(float((brightness < 0.22).mean()), 3),
        contrast=round(float(clamp(gray.std() / 128, 0, 1)), 3),
        saturation=round(float(saturation_map.mean()), 3),
        warmth=round(float((arr[..., 0].mean() - arr[..., 2].mean()) / 255), 3),
        edge_density=round(float((edge_arr > 35).mean()), 3),
        light_direction=light_direction,
        light_position=light_position,
        light_x=light_x,
        light_y=light_y,
        light_sources=light_sources,
        shadow_x=shadow_x,
        shadow_y=shadow_y,
        depth_near_mean=round(float(depth_map.mean()), 3),
        depth_near_ratio=round(float((depth_map > 0.68).mean()), 3),
        depth_complexity=round(float(depth_map.std()), 3),
        foreground_x=foreground_x,
        foreground_y=foreground_y,
        main_palette=get_palette(image),
    )

    maps = {
        "brightness": make_heatmap(brightness_blur, "light"),
        "depth": make_heatmap(depth_map, "depth"),
    }
    return analysis, maps


def make_heatmap(values: np.ndarray, mode: str) -> Image.Image:
    v = normalize(values)
    if mode == "depth":
        r = (40 + 210 * v).astype(np.uint8)
        g = (70 + 120 * (1 - np.abs(v - 0.5) * 2)).astype(np.uint8)
        b = (210 * (1 - v)).astype(np.uint8)
    else:
        r = (255 * v).astype(np.uint8)
        g = (190 * v + 30).astype(np.uint8)
        b = (70 * (1 - v)).astype(np.uint8)
    return Image.fromarray(np.dstack([r, g, b]), "RGB")


DEFAULT_IMAGE_PROMPT = "シンプルな背景紙の上にスプーンをおき、任意の場所、任意の数の光源から照明が照射されている画像を生成してください"


def optimize_generation_prompt(user_prompt: str) -> str:
    base = user_prompt.strip() or DEFAULT_IMAGE_PROMPT
    return (
        f"{base}\n\n"
        "生成条件:\n"
        "- 被写体は金属製のスプーン1本。背景は無地の背景紙とし、机上の小さな撮影セットとして成立させる。\n"
        "- 光源は1-3灯の範囲で、位置・高さ・強さが互いに異なるようにする。\n"
        "- 各光源の影響が解析できるように、ハイライト、落ち影、反射、明暗のグラデーションを明確に残す。\n"
        "- 光源そのものは画面に写さず、照射結果だけで光源位置が推測できる写真にする。\n"
        "- 生成画像らしい過剰な装飾や文字、人物、余計な物体は入れない。\n"
        "- 真上すぎないカメラ位置で、背景紙・スプーン・影の奥行きが分かる構図にする。\n"
        "- 写真調、実験記録のように中立的、1024px正方形。"
    )


def optimize_stable_diffusion_prompt(user_prompt: str) -> str:
    subject = user_prompt.strip() or DEFAULT_IMAGE_PROMPT
    return (
        "a realistic studio photograph of one stainless steel spoon placed on a simple seamless paper backdrop, "
        "minimal tabletop set, one to three off-camera light sources from different positions and heights, "
        "clear highlights on the spoon, visible cast shadows, soft gradients on the background paper, "
        "physical lighting cues, no text, no people, no extra objects, neutral documentary style, "
        "slightly angled camera view, shallow depth, high detail, natural reflections"
        f", concept: {subject}"
    )


def generate_image_with_openai(prompt: str) -> Image.Image | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    encoded = data["data"][0].get("b64_json")
    if not encoded:
        return None
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


@st.cache_resource(show_spinner=False)
def load_stable_diffusion_pipeline():
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["revision"] = "fp16"

    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", **kwargs)
    pipeline = pipeline.to(device)
    if device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline, device


def generate_image_with_stable_diffusion(prompt: str, seed: int = 42) -> Image.Image:
    import torch

    pipeline, device = load_stable_diffusion_pipeline()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        image = pipeline(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
        ).images[0]
    return image.convert("RGB")


def add_radial_light(canvas: np.ndarray, center_x: float, center_y: float, color: tuple[int, int, int], strength: float) -> None:
    h, w, _ = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius = w * (0.34 + strength * 0.16)
    falloff = np.clip(1 - dist / radius, 0, 1) ** 2
    for channel, value in enumerate(color):
        canvas[..., channel] += falloff * value * strength


def generate_local_demo_image(prompt: str, seed: int | None = None) -> Image.Image:
    rng = random.Random(seed if seed is not None else random.randint(0, 999999))
    w, h = 1024, 1024
    base = np.zeros((h, w, 3), dtype=np.float32)
    paper = np.array([168 + rng.randint(-18, 20), 160 + rng.randint(-16, 18), 145 + rng.randint(-12, 18)], dtype=np.float32)
    base[:] = paper

    light_count = rng.randint(1, 3)
    light_colors = [(255, 224, 165), (190, 218, 255), (255, 190, 210)]
    for index in range(light_count):
        cx = rng.uniform(80, 944)
        cy = rng.uniform(80, 760)
        strength = rng.uniform(0.34, 0.8) if index else rng.uniform(0.62, 0.95)
        add_radial_light(base, cx, cy, light_colors[index], strength)

    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_dx = rng.randint(35, 130)
    shadow_dy = rng.randint(35, 110)
    shadow_draw.ellipse((335 + shadow_dx, 395 + shadow_dy, 630 + shadow_dx, 675 + shadow_dy), fill=(0, 0, 0, 48))
    shadow_draw.rounded_rectangle((520 + shadow_dx, 560 + shadow_dy, 805 + shadow_dx, 625 + shadow_dy), radius=28, fill=(0, 0, 0, 42))
    shadow = shadow.filter(ImageFilter.GaussianBlur(34))
    image = Image.alpha_composite(image.convert("RGBA"), shadow)

    spoon = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(spoon)
    draw.ellipse((315, 355, 615, 650), fill=(190, 195, 196, 255), outline=(235, 238, 238, 210), width=7)
    draw.ellipse((375, 420, 560, 590), fill=(132, 138, 142, 160))
    draw.rounded_rectangle((520, 525, 830, 594), radius=30, fill=(185, 190, 190, 255), outline=(235, 238, 238, 190), width=6)
    draw.line((380, 425, 540, 540, 800, 555), fill=(255, 255, 255, 145), width=10)
    spoon = spoon.rotate(rng.uniform(-18, 18), resample=Image.Resampling.BICUBIC, center=(512, 512))
    image = Image.alpha_composite(image, spoon).convert("RGB")
    return image.filter(ImageFilter.GaussianBlur(0.6))


def classify_scene(analysis: Analysis) -> dict[str, str]:
    if analysis.edge_density > 0.22:
        object_type = "断片的な素材 / 紙片 / 枝 / 細いプロップ"
        construction = "透明糸・ピン・細い支柱で複数の小片を固定する"
    elif analysis.saturation < 0.18:
        object_type = "単色の物体 / 石膏 / 布 / 印刷面"
        construction = "1つの主素材を置き、折れ・影・余白を調整する"
    elif analysis.contrast > 0.45:
        object_type = "輪郭の強い立体物"
        construction = "主役プロップ1点と背景板2枚で構成する"
    else:
        object_type = "布 / 半透明シート / やわらかい素材"
        construction = "2点吊り、または細い支柱で浮遊感を作る"

    if analysis.brightness_mean < 0.35:
        background = "黒または濃いグレーのマットな背景"
    elif analysis.brightness_mean > 0.72:
        background = "白、薄いグレー、半透明アクリルの背景"
    else:
        background = "中間グレーの紙、または塗装した背景板"

    if analysis.aspect_ratio > 1.25:
        frame = "横長のミニステージ"
    elif analysis.aspect_ratio < 0.8:
        frame = "縦長の狭いブース"
    else:
        frame = "箱庭状の卓上セット"

    return {
        "object_type": object_type,
        "construction": construction,
        "background": background,
        "frame": frame,
    }


def light_position_for_source(source: dict[str, Any]) -> tuple[int, int, int]:
    source_x = float(source["x"])
    source_y = float(source["y"])
    x = int(14 + source_x * 72)
    y = int(12 + source_y * 76)
    dx = 0.5 - source_x
    dy = 0.5 - source_y
    angle = int(round(np.degrees(np.arctan2(dy, dx))))
    return x, y, angle


def role_label(role: str) -> str:
    return {
        "key": "キーライト",
        "fill": "フィルライト",
        "rim": "リム/バックライト",
    }.get(role, "補助ライト")


def build_single_light(source: dict[str, Any], analysis: Analysis, index: int) -> dict[str, Any]:
    brightness_span = analysis.brightness_p90 - analysis.brightness_p10
    role = str(source.get("role", "fill"))
    source_y = float(source["y"])
    source_mean = float(source.get("mean", analysis.brightness_mean))
    source_peak = float(source.get("peak", analysis.brightness_p90))
    source_area = float(source.get("area_ratio", 0.0))
    source_depth = float(source.get("depth", analysis.depth_near_mean))

    base_power = 20 + source_mean * 52 + source_peak * 18 + brightness_span * 14 + analysis.shadow_ratio * 10
    if role == "key":
        base_power += 10
    elif role == "fill":
        base_power -= 10
    else:
        base_power -= 4
    power = int(clamp(round(base_power), 15, 92))

    distance = int(round(36 + analysis.depth_complexity * 72 + source_depth * 34 + source_area * 120))
    if role == "fill":
        distance += 10
    elif role == "rim":
        distance += 6
    distance = int(clamp(distance, 30, 95))

    height = int(round(22 + (1 - source_y) * 42 + source_depth * 14))
    if source_y > 0.62:
        height -= 16
    if role == "rim":
        height += 10
    height = int(clamp(height, 12, 78))

    softness_score = 0.42 * (1 - analysis.contrast) + 0.32 * (1 - analysis.edge_density) + 0.18 * source_area + 0.08 * analysis.highlight_ratio
    quality = "柔らかい光" if softness_score > 0.48 or role == "fill" else "硬めの光"
    diffuser = "LED前 10-15 cm にトレーシングペーパー" if quality == "柔らかい光" else "拡散なし。黒ケント紙で光を絞る"
    target_lux_min = int(round(180 + power * 10))
    target_lux_max = int(round(target_lux_min + 260 + analysis.contrast * 420))
    x, y, angle = light_position_for_source(source)
    direction, label = infer_light_label(float(source["x"]), float(source["y"]))

    return {
        "id": index + 1,
        "role": role,
        "role_label": role_label(role),
        "label": label,
        "direction": direction,
        "x": x,
        "y": y,
        "source_x": float(source["x"]),
        "source_y": float(source["y"]),
        "angle_deg": angle,
        "distance_cm": distance,
        "height_cm": height,
        "power_percent": power,
        "target_lux": f"{target_lux_min}-{target_lux_max} lx",
        "quality": quality,
        "diffuser": diffuser,
        "area_ratio": source_area,
        "peak": source_peak,
        "mean": source_mean,
        "reason": f"{role_label(role)}: 明部塊の位置=({source['x']}, {source['y']}), 面積比={source_area}, ピーク明度={source_peak}, 深度={source_depth}",
    }


def build_light_plan(analysis: Analysis) -> dict[str, Any]:
    sources = analysis.light_sources or [{
        "role": "key",
        "x": analysis.light_x,
        "y": analysis.light_y,
        "area_ratio": analysis.highlight_ratio,
        "peak": analysis.brightness_p90,
        "mean": analysis.brightness_mean,
        "depth": analysis.depth_near_mean,
    }]
    lights = [build_single_light(source, analysis, index) for index, source in enumerate(sources)]
    primary = lights[0]
    return {
        **primary,
        "lights": lights,
        "source_count": len(lights),
        "reason": {
            "brightness": f"平均明度 {analysis.brightness_mean} / 明部比率 {analysis.highlight_ratio} / 暗部比率 {analysis.shadow_ratio}",
            "depth": f"近景量 {analysis.depth_near_ratio} / 深度複雑度 {analysis.depth_complexity}",
            "position": f"明部の塊を {len(lights)} 個検出し、主光源は ({analysis.light_x}, {analysis.light_y})",
        },
        "equipment": "小型LEDパネル、三脚、トレーシングペーパー、黒ケント紙",
    }


def build_physical_inference(analysis: Analysis) -> dict[str, Any]:
    scene = classify_scene(analysis)
    light = build_light_plan(analysis)
    color_temp = "暖色寄り" if analysis.warmth > 0.07 else "寒色寄り" if analysis.warmth < -0.07 else "ニュートラル"

    wind_strength = "なし"
    if "布" in scene["object_type"] or "半透明" in scene["object_type"]:
        wind_strength = "弱い"
    if analysis.edge_density > 0.28:
        wind_strength = "ごく弱く、断続的"

    depth_layers = 2 + int(analysis.depth_complexity > 0.18) + int(analysis.depth_near_ratio > 0.2)

    return {
        "concept": {
            "title": "生成イメージの外側を構築する",
            "question": "この画像が世界から切り取られたものだとしたら、フレームの外側には何があるか。",
            "method": "明るさ・明暗分布・疑似深度・エッジ量から、存在しない物理条件を推定する。",
        },
        "image_analysis": asdict(analysis),
        "physical_setup": {
            "space": {
                "scale": "30-60 cm の卓上セット",
                "format": scene["frame"],
                "background": scene["background"],
                "depth_layers": f"{depth_layers}層の奥行き",
                "foreground_position": f"近景重心 x={analysis.foreground_x}, y={analysis.foreground_y}",
            },
            "light": {
                "source_count": f"{light['source_count']}灯",
                "direction": light["label"],
                "distance": f"被写体中心から約 {light['distance_cm']} cm",
                "height": f"床面から約 {light['height_cm']} cm",
                "angle": f"被写体に対して {light['angle_deg']}°",
                "intensity": f"{light['power_percent']}% / 目安 {light['target_lux']}",
                "quality": light["quality"],
                "color_temperature": color_temp,
                "diffuser": light["diffuser"],
                "tools": light["equipment"],
            },
            "light_sources": {
                item["role_label"]: (
                    f"{item['label']} / 距離 {item['distance_cm']} cm / 高さ {item['height_cm']} cm / "
                    f"角度 {item['angle_deg']}° / 強さ {item['power_percent']}% / {item['quality']}"
                )
                for item in light["lights"]
            },
            "object": {
                "type": scene["object_type"],
                "size": "主素材は 20-30 cm 程度",
                "setup": scene["construction"],
                "surface": "抽出色に近づける。ただし端や固定具はわずかに物理的に見せる",
            },
            "camera": {
                "angle": "ややローアングル" if analysis.foreground_y > 0.58 else "目線の高さ",
                "distance": "被写体から 45-70 cm",
                "lens_feel": "50-70 mm 相当。背景を少し圧縮する",
                "frame_rule": "元画像を内側のフレームとして固定し、その外に 20-40 cm 拡張する",
            },
            "wind_or_motion": {
                "direction": "右から左" if analysis.light_x < 0.5 else "左から右",
                "strength": wind_strength,
                "note": "動きは痕跡として扱う。完成状態では出来事が起きていないようにする",
            },
        },
        "light_plan": light,
        "outside_frame_plan": [
            "AI画像を中央フレームとして印刷、またはモニター表示する。",
            "明るさマップで見える明部の方向に、物理ライトを配置する。",
            "深度マップで近いと推定された領域を、実セットの前景に置く。",
            "床・壁・影の方向を、画像の外側へ連続させる。",
            "出来事ではなく、出来事が起こる前後のセットだけを残す。",
            "元画像と同じカメラ位置から、外側を含めて記録する。",
        ],
        "production_checklist": [
            "中央のAI画像を準備した",
            "明るさマップから光源方向を確認した",
            "深度マップから前景・背景の層を決めた",
            "ライトの距離・高さ・角度を合わせた",
            "影の方向が画像と外側でつながっている",
            "フレーム外の空間を記録した",
        ],
    }


def make_markdown_report(data: dict[str, Any]) -> str:
    setup = data["physical_setup"]
    analysis = data["image_analysis"]

    lines = [
        "# 制作ガイド",
        "",
        "## コンセプト",
        data["concept"]["question"],
        "",
        "## 画像解析",
        f"- サイズ: {analysis['width']} x {analysis['height']}",
        f"- 平均明度: {analysis['brightness_mean']}",
        f"- 明度 p10 / p50 / p90: {analysis['brightness_p10']} / {analysis['brightness_p50']} / {analysis['brightness_p90']}",
        f"- 暗部比率: {analysis['shadow_ratio']}",
        f"- 明部比率: {analysis['highlight_ratio']}",
        f"- 深度 近景量: {analysis['depth_near_ratio']}",
        f"- 深度 複雑度: {analysis['depth_complexity']}",
        f"- 推定光源: {analysis['light_position']}",
        f"- 主要色: {', '.join(analysis['main_palette'])}",
        "",
        "## 物理セット",
    ]

    labels = {
        "space": "空間",
        "light": "光",
        "light_sources": "光源一覧",
        "object": "物体",
        "camera": "カメラ",
        "wind_or_motion": "風・動き",
    }
    for group_name, group in setup.items():
        lines.append(f"### {labels.get(group_name, group_name)}")
        for key, value in group.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## フレーム外の制作手順")
    lines.extend(f"{index}. {item}" for index, item in enumerate(data["outside_frame_plan"], start=1))
    lines.append("")
    lines.append("## チェックリスト")
    lines.extend(f"- [ ] {item}" for item in data["production_checklist"])
    return "\n".join(lines)


def render_interactive_3d_scene(light: dict[str, Any], height: int = 430) -> None:
    lights = light.get("lights", [light])
    scene = {
        "lights": [
            {
                "x": round((item["x"] - 50) * 1.2, 2),
                "z": round((item["y"] - 50) * 1.2, 2),
                "y": float(item["height_cm"]),
                "distance": int(item["distance_cm"]),
                "power": int(item["power_percent"]),
                "angle": int(item["angle_deg"]),
                "lux": item["target_lux"],
                "role": item["role_label"],
                "coord": (
                    f"x={round((float(item.get('source_x', 0.5)) - 0.5) * 60)}cm, "
                    f"y={int(item['height_cm'])}cm, "
                    f"z={round((float(item.get('source_y', 0.5)) - 0.5) * 60)}cm"
                ),
            }
            for item in lights
        ],
        "distance": int(light["distance_cm"]),
        "power": int(light["power_percent"]),
        "angle": int(light["angle_deg"]),
        "lux": light["target_lux"],
    }
    scene_json = json.dumps(scene, ensure_ascii=False)
    html = """
    <div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div style="font-weight:650;color:#222;">3D配置ビュー</div>
        <div style="font-size:12px;color:#555;">ドラッグで回転 / ホイールで拡大縮小</div>
      </div>
      <canvas id="scene3d" width="920" height="430" style="width:100%;height:clamp(240px,32vw,__CANVAS_HEIGHT__px);border:1px solid #d4d0c8;border-radius:12px;background:#f7f5ef;display:block;"></canvas>
    </div>
    <script>
    const scene = __SCENE__;
    const canvas = document.getElementById("scene3d");
    const ctx = canvas.getContext("2d");
    let viewWidth = 920;
    let viewHeight = 430;
    let yaw = -0.72;
    let pitch = 0.48;
    let zoom = 4.1;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const subject = { x: 0, y: 22, z: 0, w: 24, h: 44, d: 24 };
    const camera = { x: 0, y: 16, z: 82 };
    const lights = scene.lights.map((item, index) => ({
      x: item.x,
      y: item.y,
      z: item.z,
      distance: item.distance,
      power: item.power,
      angle: item.angle,
      lux: item.lux,
      role: item.role,
      coord: item.coord,
      color: index === 0 ? "#ffd166" : index === 1 ? "#a7d8ff" : "#ff9bb2",
      beam: index === 0 ? "#f0a202" : index === 1 ? "#6aa7d8" : "#dc6680"
    }));

    function rotate(point) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = point.x * cy - point.z * sy;
      const z1 = point.x * sy + point.z * cy;
      const y1 = point.y * cp - z1 * sp;
      const z2 = point.y * sp + z1 * cp;
      return { x: x1, y: y1, z: z2 };
    }

    function project(point) {
      const r = rotate(point);
      const perspective = 1 / (1 + (r.z + 140) / 520);
      const responsiveScale = Math.max(0.46, Math.min(1.2, Math.min(viewWidth / 920, viewHeight / 430)));
      return {
        x: viewWidth / 2 + r.x * zoom * responsiveScale * perspective,
        y: viewHeight / 2 - r.y * zoom * responsiveScale * perspective,
        z: r.z,
        p: perspective
      };
    }

    function line(a, b, color, width = 1, dash = []) {
      const pa = project(a), pb = project(b);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    }

    function label(text, point, color = "#222") {
      const p = project(point);
      ctx.fillStyle = color;
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(text, p.x + 8, p.y - 8);
    }

    function labelBox(title, detail, point, color = "#222") {
      const p = project(point);
      ctx.save();
      ctx.font = "12px system-ui, sans-serif";
      const width = Math.max(ctx.measureText(title).width, ctx.measureText(detail).width) + 14;
      const x = p.x + 10;
      const y = p.y - 34;
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.strokeStyle = "rgba(40,40,40,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x, y, width, 34, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(title, x + 7, y + 13);
      ctx.fillStyle = "#333";
      ctx.fillText(detail, x + 7, y + 28);
      ctx.restore();
    }

    function sphere(point, radius, fill, stroke = "#222") {
      const p = project(point);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * p.p, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    function drawBox(center, size, fill, stroke) {
      const x = center.x, y = center.y, z = center.z;
      const w = size.w / 2, h = size.h / 2, d = size.d / 2;
      const pts = [
        {x:x-w,y:y-h,z:z-d},{x:x+w,y:y-h,z:z-d},{x:x+w,y:y+h,z:z-d},{x:x-w,y:y+h,z:z-d},
        {x:x-w,y:y-h,z:z+d},{x:x+w,y:y-h,z:z+d},{x:x+w,y:y+h,z:z+d},{x:x-w,y:y+h,z:z+d}
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const faces = [[0,1,2,3],[1,5,6,2],[4,5,6,7]];
      ctx.save();
      for (const face of faces) {
        ctx.beginPath();
        face.forEach((idx, i) => {
          const p = project(pts[idx]);
          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }
      for (const e of edges) line(pts[e[0]], pts[e[1]], stroke, 1);
      ctx.restore();
    }

    function drawGround() {
      for (let v = -80; v <= 80; v += 20) {
        line({x:-80,y:0,z:v}, {x:80,y:0,z:v}, "#d8d2c8", 1);
        line({x:v,y:0,z:-80}, {x:v,y:0,z:80}, "#d8d2c8", 1);
      }
      line({x:-88,y:0,z:0}, {x:88,y:0,z:0}, "#9d9488", 1.5);
      line({x:0,y:0,z:-88}, {x:0,y:0,z:88}, "#9d9488", 1.5);
    }

    function drawBeam(light, index) {
      const target = { x: 0, y: 25, z: 0 };
      const a = project(light), b = project(target);
      const alpha = Math.min(0.56, 0.12 + light.power / 210);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = light.beam;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      const spread = index === 1 ? 54 : 38;
      ctx.lineTo(b.x - spread * b.p, b.y + 18 * b.p);
      ctx.lineTo(b.x + spread * b.p, b.y - 18 * b.p);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      line(light, target, light.beam, index === 0 ? 3 : 2.2);
      ctx.restore();
    }

    function render() {
      ctx.clearRect(0, 0, viewWidth, viewHeight);
      drawGround();
      lights.forEach((item, index) => drawBeam(item, index));
      drawBox({ x: subject.x, y: subject.y, z: subject.z }, subject, "rgba(208,208,203,0.82)", "#222");
      lights.forEach((item) => {
        sphere(item, item.role.includes("キー") ? 8 : 6.4, item.color);
        line(item, {x:item.x,y:0,z:item.z}, "#777", 1, [3, 4]);
        line(item, {x:0,y:item.y,z:0}, "#777", 1, [3, 4]);
        labelBox(item.role, item.coord, item, item.beam);
      });
      sphere(camera, 6, "#444", "#222");
      line(camera, {x:0,y:22,z:0}, "#555", 2, [5, 4]);
      label("被写体", {x:8,y:48,z:0});
      label("カメラ", camera);

      ctx.fillStyle = "#242424";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(`検出光源 ${lights.length}灯 / 主光源: 距離 ${scene.distance}cm / 角度 ${scene.angle}° / 強さ ${scene.power}%`, 18, 26);
      ctx.fillStyle = "#555";
      ctx.fillText(`目安照度 ${scene.lux}`, 18, 46);
      lights.forEach((item, index) => {
        ctx.fillStyle = item.color;
        ctx.fillRect(18, 62 + index * 18, 10, 10);
        ctx.fillStyle = "#333";
        ctx.fillText(`${item.role}: ${item.distance}cm / ${item.y}cm / ${item.power}%`, 34, 72 + index * 18);
      });
    }

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.01;
      pitch += (event.clientY - lastY) * 0.008;
      pitch = Math.max(-1.1, Math.min(1.15, pitch));
      lastX = event.clientX;
      lastY = event.clientY;
      render();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoom += event.deltaY > 0 ? -0.25 : 0.25;
      zoom = Math.max(2.4, Math.min(7, zoom));
      render();
    }, { passive: false });

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      viewWidth = Math.max(1, rect.width);
      viewHeight = Math.max(1, rect.height);
      const nextWidth = Math.round(viewWidth * dpr);
      const nextHeight = Math.round(viewHeight * dpr);
      if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
        canvas.width = nextWidth;
        canvas.height = nextHeight;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    }

    const resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(canvas);
    resizeCanvas();
    </script>
    """.replace("__SCENE__", scene_json).replace("__CANVAS_HEIGHT__", str(height))
    components.html(html, height=height + 55)


def render_light_diagram(light: dict[str, Any]) -> None:
    subject_x = 50
    subject_y = 50
    subject_height = 44
    light_x = light["x"]
    light_y = light["y"]
    power = int(light["power_percent"])
    beam_opacity = 0.16 + power / 230
    light_height = int(clamp(86 - light["height_cm"], 18, 78))
    side_depth = int(clamp(light_y, 16, 84))

    top_svg = f"""
    <svg viewBox="0 0 100 100" width="100%" height="260" role="img" aria-label="上面図">
      <rect x="8" y="10" width="84" height="80" rx="2" fill="#f6f4ef" stroke="#3a3a3a" stroke-width="0.8"/>
      <text x="12" y="17" font-size="4" fill="#333">上面図</text>
      <rect x="34" y="36" width="32" height="28" rx="2" fill="#d9d9d9" stroke="#111" stroke-width="1.2"/>
      <text x="50" y="52" text-anchor="middle" font-size="4.2" fill="#111">被写体</text>
      <line x1="{light_x}" y1="{light_y}" x2="{subject_x}" y2="{subject_y}" stroke="#f0a202" stroke-width="2.8" stroke-linecap="round" opacity="{beam_opacity}"/>
      <polygon points="{light_x},{light_y} {subject_x - 10},{subject_y - 8} {subject_x + 10},{subject_y + 8}" fill="#f0a202" opacity="{beam_opacity / 2}"/>
      <circle cx="{light_x}" cy="{light_y}" r="6" fill="#ffd166" stroke="#2b2b2b" stroke-width="1.2"/>
      <text x="{light_x}" y="{light_y + 1.4}" text-anchor="middle" font-size="3.5" fill="#111">LED</text>
      <line x1="50" y1="92" x2="50" y2="67" stroke="#444" stroke-width="1.5" marker-end="url(#arrowTop)"/>
      <text x="50" y="97" text-anchor="middle" font-size="3.7" fill="#111">カメラ</text>
      <defs><marker id="arrowTop" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#444"/></marker></defs>
    </svg>
    """

    front_svg = f"""
    <svg viewBox="0 0 100 100" width="100%" height="260" role="img" aria-label="前面図">
      <rect x="8" y="10" width="84" height="80" rx="2" fill="#f7f7f4" stroke="#3a3a3a" stroke-width="0.8"/>
      <text x="12" y="17" font-size="4" fill="#333">前面図</text>
      <line x1="14" y1="82" x2="86" y2="82" stroke="#777" stroke-width="1"/>
      <rect x="39" y="{subject_height}" width="22" height="{82 - subject_height}" rx="2" fill="#d9d9d9" stroke="#111" stroke-width="1.2"/>
      <text x="50" y="64" text-anchor="middle" font-size="4" fill="#111">被写体</text>
      <line x1="{light_x}" y1="{light_height}" x2="50" y2="{subject_height + 14}" stroke="#f0a202" stroke-width="2.8" stroke-linecap="round" opacity="{beam_opacity}"/>
      <polygon points="{light_x},{light_height} 42,{subject_height + 10} 58,{subject_height + 18}" fill="#f0a202" opacity="{beam_opacity / 2}"/>
      <circle cx="{light_x}" cy="{light_height}" r="6" fill="#ffd166" stroke="#2b2b2b" stroke-width="1.2"/>
      <text x="{light_x}" y="{light_height + 1.4}" text-anchor="middle" font-size="3.5" fill="#111">LED</text>
      <line x1="{light_x + 8}" y1="{light_height}" x2="{light_x + 8}" y2="82" stroke="#777" stroke-dasharray="2 2" stroke-width="0.8"/>
      <text x="{light_x + 11}" y="80" font-size="3.3" fill="#333">高さ</text>
      <circle cx="50" cy="94" r="4" fill="#444"/>
      <text x="50" y="98" text-anchor="middle" font-size="3.5" fill="#111">カメラ</text>
    </svg>
    """

    side_svg = f"""
    <svg viewBox="0 0 100 100" width="100%" height="260" role="img" aria-label="側面図">
      <rect x="8" y="10" width="84" height="80" rx="2" fill="#f4f6f7" stroke="#3a3a3a" stroke-width="0.8"/>
      <text x="12" y="17" font-size="4" fill="#333">側面図</text>
      <line x1="14" y1="82" x2="86" y2="82" stroke="#777" stroke-width="1"/>
      <rect x="45" y="{subject_height}" width="14" height="{82 - subject_height}" rx="2" fill="#d9d9d9" stroke="#111" stroke-width="1.2"/>
      <text x="52" y="64" text-anchor="middle" font-size="4" fill="#111">被写体</text>
      <line x1="{side_depth}" y1="{light_height}" x2="52" y2="{subject_height + 14}" stroke="#f0a202" stroke-width="2.8" stroke-linecap="round" opacity="{beam_opacity}"/>
      <polygon points="{side_depth},{light_height} 45,{subject_height + 10} 59,{subject_height + 18}" fill="#f0a202" opacity="{beam_opacity / 2}"/>
      <circle cx="{side_depth}" cy="{light_height}" r="6" fill="#ffd166" stroke="#2b2b2b" stroke-width="1.2"/>
      <text x="{side_depth}" y="{light_height + 1.4}" text-anchor="middle" font-size="3.5" fill="#111">LED</text>
      <line x1="{side_depth}" y1="86" x2="52" y2="86" stroke="#777" stroke-dasharray="2 2" stroke-width="0.8"/>
      <text x="{(side_depth + 52) / 2}" y="91" text-anchor="middle" font-size="3.3" fill="#333">距離</text>
      <polygon points="50,94 45,99 55,99" fill="#444"/>
      <text x="50" y="98" text-anchor="middle" font-size="3.5" fill="#111">カメラ</text>
    </svg>
    """

    views = st.columns(3)
    views[0].markdown(top_svg, unsafe_allow_html=True)
    views[1].markdown(front_svg, unsafe_allow_html=True)
    views[2].markdown(side_svg, unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("主光源距離", f"{light['distance_cm']} cm")
    cols[1].metric("主光源高さ", f"{light['height_cm']} cm")
    cols[2].metric("主光源角度", f"{light['angle_deg']}°")
    cols[3].metric("光源数", f"{light.get('source_count', 1)}灯")
    st.caption(f"目安照度: {light['target_lux']} / 光質: {light['quality']} / {light['diffuser']}")

    st.write("検出された光源")
    st.dataframe(
        [
            {
                "役割": item["role_label"],
                "方向": item["label"],
                "距離cm": item["distance_cm"],
                "高さcm": item["height_cm"],
                "角度": item["angle_deg"],
                "強さ%": item["power_percent"],
                "光質": item["quality"],
            }
            for item in light.get("lights", [light])
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("推定の根拠"):
        for value in light["reason"].values():
            st.write(value)


def render_palette(colors: list[str]) -> None:
    cols = st.columns(len(colors) if colors else 1)
    for col, color in zip(cols, colors):
        col.markdown(
            f"""
            <div style="height:44px;background:{color};border:1px solid rgba(0,0,0,.18);border-radius:4px"></div>
            <small>{color}</small>
            """,
            unsafe_allow_html=True,
        )


def render_metric(label: str, value: float) -> None:
    st.metric(label, f"{value:.2f}")
    st.progress(clamp(value, 0, 1))


def apply_design_system() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17211d;
            --muted: #66736d;
            --line: #dce4df;
            --surface: #ffffff;
            --surface-soft: #f4f7f5;
            --accent: #1f6f54;
            --accent-soft: #e4f2ec;
            --warm: #e69b45;
        }
        .stApp { background: #f7f9f8; color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            width: 100%;
            max-width: none;
            padding: clamp(.75rem, 1.6vw, 1.6rem) clamp(.75rem, 2.2vw, 3rem) 4rem;
        }
        [data-testid="stMainBlockContainer"] > div { width: 100%; max-width: none; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
        h1 { font-size: clamp(1.7rem, 2.5vw, 2.8rem) !important; line-height: 1.03 !important; margin-bottom: .35rem !important; }
        h2 { font-size: clamp(1.05rem, 1.35vw, 1.35rem) !important; margin-top: 0.08rem !important; }
        h3 { font-size: 1.08rem !important; }
        p, label, .stCaption { color: var(--muted); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--line) !important;
            border-radius: 20px !important;
            box-shadow: 0 12px 34px rgba(31, 55, 45, 0.06);
        }
        [data-testid="stImage"] img { border-radius: 14px; }
        [data-testid="stMetric"] {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.35rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid var(--line);
            min-height: 2.75rem;
            font-weight: 650;
        }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface-soft);
            border: 1.5px dashed #aebdb5;
            border-radius: 16px;
        }
        [data-baseweb="tab-list"] { gap: 0.45rem; }
        [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.4rem 0.95rem;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            color: var(--accent);
            background: var(--accent-soft);
            border-radius: 999px;
            padding: .35rem .75rem;
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .section-kicker { color: var(--accent); font-weight: 750; font-size: .82rem; letter-spacing: .06em; }
        .hero-copy { max-width: 760px; font-size: 1.05rem; line-height: 1.75; color: var(--muted); }
        .panel-note {
            background: var(--surface-soft);
            border-left: 3px solid var(--accent);
            border-radius: 0 12px 12px 0;
            padding: .85rem 1rem;
            color: var(--muted);
            font-size: .9rem;
        }
        .result-header {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin: .2rem 0 .65rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.result-window-marker) {
            border: 1px solid rgba(31,111,84,.42) !important;
            box-shadow: 0 20px 54px rgba(31, 74, 56, .14);
            background: linear-gradient(180deg, #ffffff 0%, #fbfdfc 100%);
        }
        .result-window-marker { display: none; }
        .result-header p { margin: 0; }
        .compact-meta { font-size: .8rem; color: var(--muted); }
        .analysis-map img {
            width: 100%;
            height: clamp(110px, 15vh, 160px);
            object-fit: cover;
            border-radius: 10px;
        }
        [data-testid="stMetric"] { min-width: 0; }
        [data-testid="stMetricLabel"] { font-size: .78rem; }
        [data-testid="stMetricValue"] { font-size: clamp(1rem, 1.25vw, 1.35rem); }
        @media (min-width: 1200px) {
            [data-testid="stHorizontalBlock"] { gap: clamp(.75rem, 1.2vw, 1.5rem); }
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding-inline: .7rem; }
            .result-header { align-items: start; flex-direction: column; }
            h1 { font-size: 1.7rem !important; }
        }
        .step-number {
            display: inline-grid;
            place-items: center;
            width: 1.65rem;
            height: 1.65rem;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            font-size: .8rem;
            font-weight: 750;
            margin-right: .45rem;
        }
        hr { border-color: var(--line) !important; margin: 2.8rem 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_light_panel(data: dict[str, Any]) -> None:
    light = data["light_plan"]
    with st.container(border=True):
        st.markdown('<div class="section-kicker">LIGHTING PLAN</div>', unsafe_allow_html=True)
        st.subheader("推定ライト配置")
        st.caption("ドラッグで回転、ホイールで拡大・縮小できます。")

        summary = st.columns(4)
        summary[0].metric("光源", f"{light['source_count']}灯")
        summary[1].metric("距離", f"{light['distance_cm']} cm")
        summary[2].metric("高さ", f"{light['height_cm']} cm")
        summary[3].metric("出力", f"{light['power_percent']}%")

        render_interactive_3d_scene(light)
        st.markdown(
            f'<div class="panel-note"><strong>{light["label"]}</strong><br>'
            f'角度 {light["angle_deg"]}° ・ 目安 {light["target_lux"]} ・ {light["quality"]}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("平面・正面・側面図を見る"):
            render_light_diagram(light)


def render_analysis_panel(data: dict[str, Any], maps: dict[str, Image.Image]) -> None:
    analysis = data["image_analysis"]

    st.markdown('<div class="section-kicker">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    st.subheader("画像解析結果")
    st.caption("明暗・奥行き・色の特徴を、物理セットへ変換するための観測値です。")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("平均明度", float(analysis["brightness_mean"]))
    with metric_cols[1]:
        render_metric("暗部比率", float(analysis["shadow_ratio"]))
    with metric_cols[2]:
        render_metric("近景量", float(analysis["depth_near_ratio"]))
    with metric_cols[3]:
        render_metric("深度複雑度", float(analysis["depth_complexity"]))

    map_cols = st.columns(2, gap="large")
    with map_cols[0]:
        with st.container(border=True):
            st.markdown("**明るさマップ**")
            st.caption("明るい領域ほど黄白で表示")
            st.image(maps["brightness"], use_container_width=True)
    with map_cols[1]:
        with st.container(border=True):
            st.markdown("**疑似深度マップ**")
            st.caption("近い領域ほど赤、遠い領域ほど青で表示")
            st.image(maps["depth"], use_container_width=True)

    with st.container(border=True):
        st.markdown("**抽出された主要色**")
        render_palette(analysis["main_palette"])


def render_production_guide(data: dict[str, Any]) -> None:
    setup = data["physical_setup"]
    st.markdown('<div class="section-kicker">PRODUCTION GUIDE</div>', unsafe_allow_html=True)
    st.subheader("制作ガイド")
    st.caption("解析値を、実際の撮影環境と制作手順へ落とし込みます。")

    tabs = st.tabs(["空間", "主光源", "光源一覧", "物体", "カメラ", "風・動き", "制作手順"])
    with tabs[0]:
        st.json(setup["space"])
    with tabs[1]:
        st.json(setup["light"])
    with tabs[2]:
        st.json(setup["light_sources"])
    with tabs[3]:
        st.json(setup["object"])
    with tabs[4]:
        st.json(setup["camera"])
    with tabs[5]:
        st.json(setup["wind_or_motion"])
    with tabs[6]:
        for index, step in enumerate(data["outside_frame_plan"], start=1):
            st.markdown(f'<p><span class="step-number">{index}</span>{step}</p>', unsafe_allow_html=True)


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_contained_image(image: Image.Image, alt: str, max_height: int = 360) -> None:
    st.markdown(
        f"""
        <div style="display:grid;place-items:center;width:100%;height:clamp(250px,42vh,{max_height}px);"
             role="img" aria-label="{alt}">
          <img src="{image_data_url(image)}" alt="{alt}"
               style="display:block;max-width:100%;max-height:100%;object-fit:contain;border-radius:12px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_analysis(data: dict[str, Any], maps: dict[str, Image.Image]) -> None:
    analysis = data["image_analysis"]
    st.markdown('<div class="section-kicker">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    st.subheader("解析結果")

    metrics = st.columns(4)
    metrics[0].metric("平均明度", f"{float(analysis['brightness_mean']):.2f}")
    metrics[1].metric("暗部比率", f"{float(analysis['shadow_ratio']):.2f}")
    metrics[2].metric("近景量", f"{float(analysis['depth_near_ratio']):.2f}")
    metrics[3].metric("深度複雑度", f"{float(analysis['depth_complexity']):.2f}")

    map_cols = st.columns([1, 1, .75], gap="medium")
    with map_cols[0]:
        st.caption("明るさマップ")
        st.image(maps["brightness"], use_container_width=True)
    with map_cols[1]:
        st.caption("疑似深度マップ")
        st.image(maps["depth"], use_container_width=True)
    with map_cols[2]:
        st.caption("主要色")
        render_palette(analysis["main_palette"])
        st.markdown(
            f'<div class="panel-note"><strong>推定光源</strong><br>{analysis["light_position"]}</div>',
            unsafe_allow_html=True,
        )


def render_compact_light(data: dict[str, Any]) -> None:
    light = data["light_plan"]
    st.markdown('<div class="section-kicker">LIGHTING PLAN</div>', unsafe_allow_html=True)
    st.subheader("照明配置図")
    summary = st.columns(4)
    summary[0].metric("光源", f"{light['source_count']}灯")
    summary[1].metric("距離", f"{light['distance_cm']}cm")
    summary[2].metric("高さ", f"{light['height_cm']}cm")
    summary[3].metric("出力", f"{light['power_percent']}%")
    render_interactive_3d_scene(light, height=330)


def render_result_dashboard(
    image: Image.Image,
    source_label: str,
    analysis: Analysis,
    result: dict[str, Any],
    maps: dict[str, Image.Image],
) -> None:
    with st.container(border=True):
        st.markdown('<span class="result-window-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-header">
              <div><span class="eyebrow">ANALYSIS COMPLETE</span></div>
              <p class="compact-meta">{analysis.width} × {analysis.height} px ／ 縦横比 {analysis.aspect_ratio}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        image_col, light_col = st.columns([0.9, 1.35], gap="large")
        with image_col:
            st.markdown('<div class="section-kicker">SOURCE IMAGE</div>', unsafe_allow_html=True)
            st.subheader(source_label)
            render_contained_image(image, source_label, max_height=390)
        with light_col:
            render_compact_light(result)

        st.divider()
        render_compact_analysis(result, maps)


def legacy_main() -> None:
    apply_design_system()
    st.markdown('<span class="eyebrow">Image → Physical Setup</span>', unsafe_allow_html=True)
    st.title("Image to Physical Setup")
    # st.markdown(
    #     '<p class="hero-copy">一枚の画像から光・奥行き・色を読み取り、撮影可能なライト配置と物理セットへ翻訳します。</p>',
    #     unsafe_allow_html=True,
    # )

    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
    if "optimized_prompt" not in st.session_state:
        st.session_state.optimized_prompt = optimize_generation_prompt(DEFAULT_IMAGE_PROMPT)

    with st.container(border=True):
        input_mode = st.radio(
            "入力方法",
            ["画像をアップロード", "サイト内で生成"],
            horizontal=True,
            label_visibility="collapsed",
        )

    image: Image.Image | None = None
    source_label = ""

    if input_mode == "サイト内で生成":
        with st.container(border=True):
            st.markdown('<div class="section-kicker">GENERATE</div>', unsafe_allow_html=True)
            st.subheader("検証用の画像を生成")
            user_prompt = st.text_area(
                "生成したいイメージ",
                value=DEFAULT_IMAGE_PROMPT,
                height=92,
            )
            optimized_prompt = optimize_generation_prompt(user_prompt)
            sd_prompt = optimize_stable_diffusion_prompt(user_prompt)
            st.session_state.optimized_prompt = optimized_prompt
            with st.expander("最適化されたプロンプト"):
                st.caption("OpenAI / 汎用")
                st.write(optimized_prompt)
                st.caption("Stable Diffusion v1-4向け")
                st.write(sd_prompt)

            generate_cols = st.columns(3)
            if generate_cols[0].button("デモ画像を生成", use_container_width=True, type="primary"):
                st.session_state.generated_image = generate_local_demo_image(optimized_prompt)
            if generate_cols[1].button("Stable Diffusionで生成", use_container_width=True):
                try:
                    with st.spinner("Stable Diffusionで画像を生成しています。初回はモデル取得に時間がかかります..."):
                        st.session_state.generated_image = generate_image_with_stable_diffusion(sd_prompt, seed=42)
                except ModuleNotFoundError:
                    st.error(
                        "Stable Diffusion用ライブラリが未インストールです。"
                        " `pip install diffusers transformers ftfy accelerate torch` を実行してください。"
                    )
                except Exception as exc:
                    st.error(f"Stable Diffusion生成に失敗しました: {exc}")
            if generate_cols[2].button("OpenAIで画像生成", use_container_width=True):
                try:
                    with st.spinner("画像を生成しています..."):
                        generated = generate_image_with_openai(optimized_prompt)
                    if generated is None:
                        st.warning("OPENAI_API_KEY が設定されていないため、OpenAI生成は使えません。ローカルデモ生成を使ってください。")
                    else:
                        st.session_state.generated_image = generated
                except Exception as exc:
                    st.error(f"画像生成に失敗しました: {exc}")

        image = st.session_state.generated_image
        source_label = "生成画像"
    else:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">UPLOAD</div>', unsafe_allow_html=True)
            st.subheader("解析する画像を選択")
            uploaded_file = st.file_uploader(
                "画像ファイル",
                type=["png", "jpg", "jpeg", "webp"],
                help="PNG / JPG / JPEG / WEBP に対応しています。",
                label_visibility="collapsed",
            )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            source_label = "入力画像"

    if image is None:
        st.info("画像をアップロードすると、入力画像・ライト配置・解析結果がここに表示されます。")
        return

    analysis, maps = analyze_image(image)
    result = build_physical_inference(analysis)
    markdown_report = make_markdown_report(result)
    json_report = json.dumps(result, ensure_ascii=False, indent=2)

    st.divider()
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">SOURCE IMAGE</div>', unsafe_allow_html=True)
            st.subheader(source_label)
            st.image(image, use_container_width=True)
            st.caption(f"{analysis.width} × {analysis.height} px　／　縦横比 {analysis.aspect_ratio}")
    with right:
        render_light_panel(result)

    st.divider()
    render_analysis_panel(result, maps)

    st.divider()
    render_production_guide(result)

    st.divider()
    st.markdown('<div class="section-kicker">EXPORT</div>', unsafe_allow_html=True)
    st.subheader("設計データを書き出す")
    export_cols = st.columns(2)
    export_cols[0].download_button(
        "セット指示JSONをダウンロード",
        data=json_report,
        file_name="physical_setup.json",
        mime="application/json",
    )
    export_cols[1].download_button(
        "制作ガイドをダウンロード",
        data=markdown_report,
        file_name="production_guide.md",
        mime="text/markdown",
    )

    with st.expander("制作ガイドの内容を確認"):
        st.markdown(markdown_report)


def previous_main() -> None:
    apply_design_system()
    st.markdown('<span class="eyebrow">Image → Physical Setup</span>', unsafe_allow_html=True)
    st.title("Image to Physical Setup")

    for key in ("generated_image", "uploaded_image"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "optimized_prompt" not in st.session_state:
        st.session_state.optimized_prompt = optimize_generation_prompt(DEFAULT_IMAGE_PROMPT)

    selected_mode = st.session_state.get("input_mode_v2", "画像をアップロード")
    existing_image = (
        st.session_state.uploaded_image
        if selected_mode == "画像をアップロード"
        else st.session_state.generated_image
    )

    with st.expander(
        "入力画像を選択" if existing_image is None else "入力画像を変更",
        expanded=existing_image is None,
    ):
        input_mode = st.radio(
            "入力方法",
            ["画像をアップロード", "サイト内で生成"],
            horizontal=True,
            key="input_mode_v2",
            label_visibility="collapsed",
        )

        if input_mode == "画像をアップロード":
            uploaded_file = st.file_uploader(
                "画像ファイル",
                type=["png", "jpg", "jpeg", "webp"],
                help="PNG / JPG / JPEG / WEBP に対応しています。",
                label_visibility="collapsed",
                key="upload_v2",
            )
            if uploaded_file is not None:
                st.session_state.uploaded_image = Image.open(uploaded_file).copy()
        else:
            user_prompt = st.text_area(
                "生成したいイメージ",
                value=DEFAULT_IMAGE_PROMPT,
                height=76,
                key="prompt_v2",
            )
            optimized_prompt = optimize_generation_prompt(user_prompt)
            sd_prompt = optimize_stable_diffusion_prompt(user_prompt)
            st.session_state.optimized_prompt = optimized_prompt

            generate_cols = st.columns(3)
            if generate_cols[0].button("デモ画像を生成", use_container_width=True, type="primary", key="demo_v2"):
                st.session_state.generated_image = generate_local_demo_image(optimized_prompt)
            if generate_cols[1].button("Stable Diffusion", use_container_width=True, key="sd_v2"):
                try:
                    with st.spinner("画像を生成しています..."):
                        st.session_state.generated_image = generate_image_with_stable_diffusion(sd_prompt, seed=42)
                except ModuleNotFoundError:
                    st.error("Stable Diffusion用ライブラリが未インストールです。")
                except Exception as exc:
                    st.error(f"Stable Diffusion生成に失敗しました: {exc}")
            if generate_cols[2].button("OpenAI", use_container_width=True, key="openai_v2"):
                try:
                    with st.spinner("画像を生成しています..."):
                        generated = generate_image_with_openai(optimized_prompt)
                    if generated is None:
                        st.warning("OPENAI_API_KEY が設定されていません。")
                    else:
                        st.session_state.generated_image = generated
                except Exception as exc:
                    st.error(f"画像生成に失敗しました: {exc}")

    input_mode = st.session_state.get("input_mode_v2", "画像をアップロード")
    if input_mode == "画像をアップロード":
        image = st.session_state.uploaded_image
        source_label = "入力画像"
    else:
        image = st.session_state.generated_image
        source_label = "生成画像"

    if image is None:
        st.info("画像を選択すると、入力画像・照明配置図・解析結果を一画面で表示します。")
        return

    analysis, maps = analyze_image(image)
    result = build_physical_inference(analysis)
    markdown_report = make_markdown_report(result)
    json_report = json.dumps(result, ensure_ascii=False, indent=2)

    render_result_dashboard(image, source_label, analysis, result, maps)

    with st.expander("詳細解析・制作ガイド・書き出し"):
        render_analysis_panel(result, maps)
        st.divider()
        render_production_guide(result)
        st.divider()
        export_cols = st.columns(2)
        export_cols[0].download_button(
            "セット指示JSONをダウンロード",
            data=json_report,
            file_name="physical_setup.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "制作ガイドをダウンロード",
            data=markdown_report,
            file_name="production_guide.md",
            mime="text/markdown",
            use_container_width=True,
        )


def scroll_page_to_top() -> None:
    components.html(
        """
        <script>
        window.requestAnimationFrame(() => {
          const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
          if (app) {
            app.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
          } else {
            window.parent.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
          }
        });
        </script>
        """,
        height=0,
    )


def main() -> None:
    apply_design_system()
    st.markdown('<span class="eyebrow">Image → Physical Setup</span>', unsafe_allow_html=True)
    st.title("Image to Physical Setup")

    if st.session_state.pop("scroll_to_result_top", False):
        scroll_page_to_top()

    defaults = {
        "generated_image": None,
        "uploaded_image": None,
        "uploaded_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "optimized_prompt" not in st.session_state:
        st.session_state.optimized_prompt = optimize_generation_prompt(DEFAULT_IMAGE_PROMPT)

    selected_mode = st.session_state.get("input_mode_v2", "画像をアップロード")
    image = (
        st.session_state.uploaded_image
        if selected_mode == "画像をアップロード"
        else st.session_state.generated_image
    )
    source_label = "入力画像" if selected_mode == "画像をアップロード" else "生成画像"

    analysis: Analysis | None = None
    maps: dict[str, Image.Image] | None = None
    result: dict[str, Any] | None = None
    markdown_report = ""
    json_report = ""

    # 結果がある場合は、入力UIより先に画面最上部へ表示する。
    if image is not None:
        analysis, maps = analyze_image(image)
        result = build_physical_inference(analysis)
        markdown_report = make_markdown_report(result)
        json_report = json.dumps(result, ensure_ascii=False, indent=2)
        render_result_dashboard(image, source_label, analysis, result, maps)

    with st.expander(
        "入力画像を選択" if image is None else "入力画像を変更",
        expanded=image is None,
    ):
        input_mode = st.radio(
            "入力方法",
            ["画像をアップロード", "サイト内で生成"],
            horizontal=True,
            key="input_mode_v2",
            label_visibility="collapsed",
        )

        if input_mode == "画像をアップロード":
            uploaded_file = st.file_uploader(
                "画像ファイル",
                type=["png", "jpg", "jpeg", "webp"],
                help="PNG / JPG / JPEG / WEBP に対応しています。",
                label_visibility="collapsed",
                key="upload_v2",
            )
            if uploaded_file is not None:
                signature = (uploaded_file.name, uploaded_file.size)
                if signature != st.session_state.uploaded_signature:
                    st.session_state.uploaded_image = Image.open(uploaded_file).copy()
                    st.session_state.uploaded_signature = signature
                    st.session_state.scroll_to_result_top = True
                    st.rerun()
        else:
            user_prompt = st.text_area(
                "生成したいイメージ",
                value=DEFAULT_IMAGE_PROMPT,
                height=76,
                key="prompt_v2",
            )
            optimized_prompt = optimize_generation_prompt(user_prompt)
            sd_prompt = optimize_stable_diffusion_prompt(user_prompt)
            st.session_state.optimized_prompt = optimized_prompt

            generate_cols = st.columns(3)
            if generate_cols[0].button("デモ画像を生成", use_container_width=True, type="primary", key="demo_v2"):
                st.session_state.generated_image = generate_local_demo_image(optimized_prompt)
                st.session_state.scroll_to_result_top = True
                st.rerun()
            if generate_cols[1].button("Stable Diffusion", use_container_width=True, key="sd_v2"):
                try:
                    with st.spinner("画像を生成しています..."):
                        st.session_state.generated_image = generate_image_with_stable_diffusion(sd_prompt, seed=42)
                    st.session_state.scroll_to_result_top = True
                    st.rerun()
                except ModuleNotFoundError:
                    st.error("Stable Diffusion用ライブラリが未インストールです。")
                except Exception as exc:
                    st.error(f"Stable Diffusion生成に失敗しました: {exc}")
            if generate_cols[2].button("OpenAI", use_container_width=True, key="openai_v2"):
                try:
                    with st.spinner("画像を生成しています..."):
                        generated = generate_image_with_openai(optimized_prompt)
                    if generated is None:
                        st.warning("OPENAI_API_KEY が設定されていません。")
                    else:
                        st.session_state.generated_image = generated
                        st.session_state.scroll_to_result_top = True
                        st.rerun()
                except Exception as exc:
                    st.error(f"画像生成に失敗しました: {exc}")

    if image is None or analysis is None or maps is None or result is None:
        st.info("画像を選択すると、結果ウィンドウを画面上部に表示します。")
        return

    with st.expander("詳細な制作ガイド・書き出し"):
        render_production_guide(result)
        st.divider()
        export_cols = st.columns(2)
        export_cols[0].download_button(
            "セット指示JSONをダウンロード",
            data=json_report,
            file_name="physical_setup.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "制作ガイドをダウンロード",
            data=markdown_report,
            file_name="production_guide.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
