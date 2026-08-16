import os
import uuid
import itertools
import numpy as np
from PIL import Image


if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

import torch
from transformers import CLIPProcessor, CLIPModel

_clip_model = None
_clip_processor = None
CLOTHING_LABELS = [
    "a t-shirt", "a dress shirt", "a blouse", "a sweater", "a hoodie",
    "jeans", "trousers", "shorts", "a skirt", "leggings",
    "a jacket", "a coat", "a blazer",
    "sneakers", "boots", "sandals", "dress shoes",
    "a handbag", "a hat", "a scarf", "sunglasses",
    "a dress", "a jumpsuit", "overalls"
]


CATEGORY_MAPPING = {
    "top": ["t-shirt", "jersey", "cardigan", "suit", "vest", "poncho", "sweatshirt", "kimono", "shirt", "blouse", "sweater"],
    "bottom": ["jean", "skirt", "short", "pajama", "trouser", "pants", "miniskirt"],
    "outerwear": ["coat", "jacket", "cloak", "trench coat", "parka", "windbreaker"],
    "shoes": ["shoe", "boot", "sandal", "sneaker", "clog", "loafer", "running shoe"],
    "accessory": ["bag", "purse", "hat", "cap", "belt", "tie", "scarf", "glove", "sunglasses", "watch", "backpack"],
    "full": ["dress", "gown", "overalls", "academic gown", "velvet", "jumpsuit"]
}

COLOR_BUCKETS = {
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "grey":   (128, 128, 128),
    "navy":   (30,  50,  120),
    "beige":  (210, 190, 155),
    "brown":  (120, 70,  40),
    "red":    (200, 30,  30),
    "blue":   (50,  100, 200),
    "green":  (50,  140, 60),
    "yellow": (230, 210, 40),
    "orange": (230, 120, 30),
    "pink":   (230, 130, 160),
    "purple": (120, 50,  160),
}

NEUTRAL_COLORS = {"white", "black", "grey", "navy", "beige", "brown"}

def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        torch.set_num_threads(1)
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=True,
        )
        _clip_model.eval()
        _clip_model = torch.quantization.quantize_dynamic(
            _clip_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

USE_CLIP = os.environ.get("USE_CLIP", "1") == "1"

# Minimum free system memory (MB) required before attempting to load CLIP.
# Loading the full fp32 checkpoint briefly needs ~600-700MB before it gets
# quantized down, so we require solid headroom above that. On Render's
# free tier (512MB total) this check will fail and classification falls
# back to "unknown" gracefully, instead of risking an OOM kill of the
# whole container. On a normal local machine/Docker Desktop (multiple GB
# available) this passes and full CLIP classification runs.
MIN_FREE_MB_FOR_CLIP = 900

def _cgroup_available_mb():
    """Return free memory (MB) as seen by the container's cgroup limit,
    or None if no cgroup limit is set/detectable. This is what actually
    matters on Render/Docker, since /proc/meminfo reports the HOST
    machine's memory, not the container's enforced limit."""
    try:
        # cgroup v2
        with open("/sys/fs/cgroup/memory.max") as f:
            limit_raw = f.read().strip()
        if limit_raw != "max":
            limit = int(limit_raw)
            with open("/sys/fs/cgroup/memory.current") as f:
                usage = int(f.read().strip())
            return (limit - usage) / (1024 * 1024)
    except Exception:
        pass
    try:
        # cgroup v1
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
        if limit < (1 << 40):  # ignore "unlimited" sentinel values
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                usage = int(f.read().strip())
            return (limit - usage) / (1024 * 1024)
    except Exception:
        pass
    return None

def _available_memory_mb():
    """Return free memory in MB, preferring the container's actual cgroup
    limit (accurate on Render/Docker) and falling back to /proc/meminfo's
    host-level figure, or None if neither is readable (e.g. on Windows)."""
    cgroup_mb = _cgroup_available_mb()
    if cgroup_mb is not None:
        return cgroup_mb
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None

def classify_item(image_path):
    if not USE_CLIP:
        name = os.path.basename(image_path).lower()
        for cat, keywords in CATEGORY_MAPPING.items():
            for kw in keywords:
                if kw in name:
                    return cat, kw.capitalize()
        return "top", "Item"

    if _clip_model is None:
        free_mb = _available_memory_mb()
        if free_mb is not None and free_mb < MIN_FREE_MB_FOR_CLIP:
            print(f"Skipping CLIP load: only {free_mb:.0f}MB free (need {MIN_FREE_MB_FOR_CLIP}MB)")
            return "unknown", "Item"

    try:
        model, processor = get_clip()
        img = Image.open(image_path).convert("RGB")
        inputs = processor(text=CLOTHING_LABELS, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        label = CLOTHING_LABELS[best_idx].replace("a ", "").replace("an ", "")
        
        # map to category
        for cat, keywords in CATEGORY_MAPPING.items():
            if any(kw in label for kw in keywords):
                return cat, label.capitalize()
        return "unknown", label.capitalize()
    except Exception as e:
        print(f"CLIP Classification failed: {e}")
        return "unknown", "Item"

def extract_palette(image_path, n_colors=3):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    palette = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
    return palette

def saturation(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    return (mx - mn) / mx if mx != 0 else 0

def lightness(rgb):
    return sum(rgb) / (3 * 255)

def pick_accent_and_base(palette):
    accent = max(palette, key=saturation)
    base = min(palette, key=saturation)
    return accent, base

def nearest_color_name(rgb):
    return min(COLOR_BUCKETS, key=lambda name: sum((a - b)**2 for a, b in zip(rgb, COLOR_BUCKETS[name])))

from color_utils import rgb_to_lab, delta_e_cie2000

def delta_e(rgb1, rgb2):
    return delta_e_cie2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))

def pairwise_avg_delta_e(items):
    colors = [item["accent_color"] for item in items]
    pairs = list(itertools.combinations(colors, 2))
    if not pairs:
        return 0
    scores = [delta_e(a, b) for a, b in pairs]
    return sum(scores) / len(scores)

def harmony_score_from_delta_e(avg_de):
    if 15 <= avg_de <= 35:
        return 1.0
    elif 10 <= avg_de < 15 or 35 < avg_de <= 50:
        return 0.65
    elif avg_de < 10:
        return 0.4
    else:
        return 0.2

def capsule_multiplier(items):
    multiplier = 1.0
    color_names = [item["color_name"] for item in items]
    accent_names = [nearest_color_name(item["accent_color"]) for item in items]


    if any(name in NEUTRAL_COLORS for name in color_names):
        multiplier += 0.15


    non_neutral_accents = [c for c in accent_names if c not in NEUTRAL_COLORS]
    if len(set(non_neutral_accents)) > 2:
        multiplier -= 0.25


    if len(color_names) >= 2 and color_names[0] == color_names[1]:
        multiplier -= 0.2

    return max(multiplier, 0.1)

def final_score(items):
    avg_de = pairwise_avg_delta_e(items)
    base = harmony_score_from_delta_e(avg_de)
    multiplier = capsule_multiplier(items)
    return round(base * multiplier, 4)

def score_label(score):
    if score >= 0.85:
        return "Strong Match"
    elif score >= 0.60:
        return "Good Match"
    else:
        return "Acceptable"

def skin_tone_boost(combo_items, recommended_hex_list, avoid_hex_list):
    """Boost/penalize combo scores based on skin-tone palette compatibility.
    
    Converts palette hex codes to the nearest 13-color bucket name and checks
    if combo item colors appear in recommended or avoid lists.
    """
    def hex_to_rgb_tuple(hex_code):
        h = hex_code.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    rec_names = set()
    for entry in recommended_hex_list:
        hex_code = entry if isinstance(entry, str) else entry.get("hex", "")
        if hex_code:
            rec_names.add(nearest_color_name(hex_to_rgb_tuple(hex_code)))
    
    avoid_names = set()
    for entry in avoid_hex_list:
        hex_code = entry if isinstance(entry, str) else entry.get("hex", "")
        if hex_code:
            avoid_names.add(nearest_color_name(hex_to_rgb_tuple(hex_code)))
    
    boost = 0.0
    for item in combo_items:
        cname = item.get("color_name", "")
        if cname in rec_names:
            boost += 0.1
        if cname in avoid_names:
            boost -= 0.15
    return boost

def generate_combinations(wardrobe: dict) -> list:
    """

    """
    tops = wardrobe.get("top", [])
    bottoms = wardrobe.get("bottom", [])
    fulls = wardrobe.get("full", [])
    shoes = wardrobe.get("shoes", [])
    outerwear = wardrobe.get("outerwear", [])
    accessories = wardrobe.get("accessory", [])

    base_combos = []


    if tops and bottoms:
        for t, b in itertools.product(tops, bottoms):
            base_combos.append([t, b])
    
    if fulls:
        for f in fulls:
            base_combos.append([f])
            

    if not base_combos:
        for cat in ["top", "bottom", "outerwear"]:
            for item in wardrobe.get(cat, []):
                base_combos.append([item])

    if not base_combos:
        return []


    combos_with_shoes = []
    if shoes:
        for combo in base_combos:
            for s in shoes:
                combos_with_shoes.append(combo + [s])
        combos = combos_with_shoes
    else:
        combos = base_combos



    final_combos = []
    for combo in combos:

        final_combos.append(combo)
        

        if outerwear:
            for o in outerwear[:2]:
                final_combos.append(combo + [o])
        
        if accessories:
            for a in accessories[:2]:
                final_combos.append(combo + [a])

    import random
    if len(final_combos) > 300:
        final_combos = random.sample(final_combos, 300)

    return final_combos

def rank_combinations(combos, top_n=6, recommended_hex_list=None, avoid_hex_list=None):
    """
    """
    if not combos:
        return []


    seen_combos = set()
    scored = []
    for combo in combos:
        item_ids = tuple(sorted([item["id"] for item in combo]))
        if item_ids in seen_combos:
            continue
        seen_combos.add(item_ids)
        
        base_score = final_score(combo)
        boost = 0.0
        if recommended_hex_list or avoid_hex_list:
            boost = skin_tone_boost(combo, recommended_hex_list or [], avoid_hex_list or [])
        score = max(0.0, min(1.0, base_score + boost))
        scored.append({
            "items": combo,
            "score": round(score, 4),
            "label": score_label(score)
        })
    

    scored.sort(key=lambda x: x["score"], reverse=True)
    

    diverse_results = []
    item_usage = {}
    

    for s in scored:
        if len(diverse_results) >= top_n: break
        if all(item_usage.get(item["id"], 0) == 0 for item in s["items"]):
            diverse_results.append(s)
            for item in s["items"]:
                item_usage[item["id"]] = item_usage.get(item["id"], 0) + 1
                

    for s in scored:
        if len(diverse_results) >= top_n: break
        if s in diverse_results: continue
        if any(item_usage.get(item["id"], 0) == 0 for item in s["items"]):
            diverse_results.append(s)
            for item in s["items"]:
                item_usage[item["id"]] = item_usage.get(item["id"], 0) + 1
                

    for s in scored:
        if len(diverse_results) >= top_n: break
        if s in diverse_results: continue
        diverse_results.append(s)
            
    return diverse_results
