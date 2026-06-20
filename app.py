import os
import json
import uuid
import hashlib
import cv2
import numpy as np
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from reference_data import MST_LAB, SEASON_LAB, get_mst_label, GROOMING_RECOMMENDATIONS
from datetime import datetime, timedelta
import wardrobe_logic
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image as PILImage
from color_utils import hex_to_rgb, hex_to_lab, rgb_to_lab, delta_e_cie2000_vec, delta_e_cie2000

app = Flask(__name__)
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('ENV') == 'production':
        raise RuntimeError("CRITICAL ERROR: SECRET_KEY environment variable is not set in production!")
    secret_key = os.urandom(24).hex()
app.config['SECRET_KEY'] = secret_key
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

storage_uri = os.environ.get("RATELIMIT_STORAGE_URI") or os.environ.get("REDIS_URL") or "memory://"
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=storage_uri
)

@app.before_request
def check_session_timeout():
    if current_user.is_authenticated:
        last_active = session.get('last_active')
        if last_active:
            last_active_time = datetime.fromtimestamp(last_active)
            if datetime.now() - last_active_time > timedelta(minutes=30):
                logout_user()
                session.pop('last_active', None)
                flash('Your session has expired.', 'error')
                return redirect(url_for('login'))
        session['last_active'] = datetime.now().timestamp()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///slayr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_WARDROBE_ITEMS = 50

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(filepath):
    """Validate that a saved file is a genuine image. Removes the file if invalid."""
    try:
        with PILImage.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def secure_upload_filename(original_filename):
    """Generate a UUID-based filename preserving the original extension."""
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
    return f"{uuid.uuid4().hex}.{ext}"

MODEL_PATH = 'face_landmarker.task'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    full_name = db.Column(db.String(150), nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    password = db.Column(db.String(150), nullable=False)

    face_shape = db.Column(db.String(50))
    skin_hex = db.Column(db.String(20))
    undertone = db.Column(db.String(20))
    season = db.Column(db.String(50))
    mst_label = db.Column(db.String(50))

class Foundation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(100))
    product = db.Column(db.String(200))
    name = db.Column(db.String(100))
    hex_code = db.Column(db.String(20))
    l = db.Column(db.Float)
    a = db.Column(db.Float)
    b = db.Column(db.Float)
    category = db.Column(db.String(50))
    brand_tier = db.Column(db.String(50))
    hue = db.Column(db.String(50))
    image_url = db.Column(db.String(500))

class WardrobeItem(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    owner_id = db.Column(db.String(100), index=True) 
    image_path = db.Column(db.String(500))
    category = db.Column(db.String(50))
    label = db.Column(db.String(50))
    palette_json = db.Column(db.Text) 
    accent_rgb = db.Column(db.String(50)) 
    base_rgb = db.Column(db.String(50)) 
    color_name = db.Column(db.String(50))
    image_hash = db.Column(db.String(100), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserPalette(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    palette_json = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

MST_X = np.array([item[:3] for item in MST_LAB], dtype=np.float32)
SEASON_X = np.array([item[:3] for item in SEASON_LAB], dtype=np.float32)
SEASON_Y = [item[3] for item in SEASON_LAB]

RECOMMENDATIONS = {
        "Oval": "Balanced proportions. Lucky you! Most styles work.",
        "Square": "Strong jawline. Soften angles with waves or round frames.",
        "Round": "Soft angles. Add structure with angular frames or height in hair.",
        "Heart": "Wide forehead, narrow chin. Balance with chin-length bobs or full beards.",
        "Diamond": "Wide cheekbones. Highlight them, soften the chin with layered styles.",
        "Rectangle": "Long face. Add width with curls or oversized frames. Avoid extreme height.",
        "Oblong": "Long face. Add width with curls or oversized frames. Avoid extreme height.",
        "Triangle": "Wider jaw, narrow forehead. Add volume on top to balance the jaw.",
        "Pear": "Wide jaw, narrow forehead. Add volume on top to balance the jaw."
}

def analyze_face_shape(landmarks, image_width, image_height, matrix=None):
    import numpy as np

    coords = {}
    # 10/152: forehead-top/chin-tip (face height)
    # 21/251: hairline-level points (true forehead width, not temples)
    # 234/454: cheekbone/zygomatic width (unchanged - this one was correct)
    # 172/397: jaw width just below the cheekbone (true jaw line - previously
    #          58/288 sat ~100-150px above the real jawline, near mouth-corner
    #          height, which made jaw width measure artificially narrow on
    #          almost every face and collapsed the classifier into only
    #          Round/Oval outputs)
    indices = [10, 152, 234, 454, 172, 397, 21, 251, 168, 6]

    # Always use 2D pixel coordinates scaling since the facial transformation matrix 
    # maps from canonical space to world coordinates (in meters), which is incompatible 
    # with normalized coordinates and causes severe face shape deformation (e.g. Pear bias).
    for idx in indices:
        coords[idx] = np.array([landmarks[idx].x * image_width, landmarks[idx].y * image_height, 0.0])

    def dist(p1_idx, p2_idx):
        return float(np.linalg.norm(coords[p1_idx] - coords[p2_idx]))

    height = dist(10, 152)
    width_forehead = dist(21, 251)
    width_cheek = dist(234, 454)
    width_jaw = dist(172, 397)

    if height == 0:
        return {
            "primary_shape": "Oval",
            "measurements": {
                "H/W Ratio": 0.0,
                "F/J Ratio": 0.0,
                "C/J Ratio": 0.0
            }
        }

    h_w_ratio = height / width_cheek
    forehead_jaw_ratio = width_forehead / width_jaw
    cheek_jaw_ratio = width_cheek / width_jaw

    def classify(h_w, f_j, c_j):
        """
        h_w  = height / cheek_width       (face elongation)
        f_j  = forehead / jaw             (narrow vs wide base)
        c_j  = cheek / jaw                (cheek prominence vs jaw)

        Thresholds validated against the Kaggle "Face Shape Dataset" (niten19,
        5 classes x 800 images). Key findings from that validation:

        - Oblong, Round, and Square separate well on h_w / c_j (Cohen's d
          0.4-2.6 between pairs) -> thresholds below are fit to this real
          signal, not guessed.
        - Heart vs Oval do NOT separate on any 2D landmark-ratio feature we
          tested (h_w, f_j, c_j, and a 4-point jaw/chin taper profile all gave
          Cohen's d < 0.22 = negligible effect size). Manual inspection of the
          dataset showed both classes are dominated by styled hair occluding
          the jaw/cheek contour and editorial labeling of red-carpet photos,
          not a measurable bone-structure difference. This matches published
          literature where Oval is the most-confused class even for CNNs
          trained directly on pixels. Heart/Oval below is therefore a weak
          tiebreaker only, and analyze_face_shape() always attaches a
          secondary_hint when this branch fires so the result is shown as a
          close call rather than a falsely confident single label.
        """
        # ── OBLONG: clearly tallest relative to cheek width ─────────────────────
        if h_w > 1.225:
            return "Oblong"

        # ── SQUARE: minimal cheek-to-jaw taper + narrow forehead/jaw spread ──────
        if c_j < 1.24 and f_j < 1.17:
            return "Square"

        # ── ROUND: short relative to cheek width, among the remaining faces ─────
        if h_w < 1.175:
            return "Round"

        # ── WIDE JAW (triangle/pear family) ─────────────────────────────────────
        if f_j < 0.92:
            if c_j < 1.02:
                return "Triangle"
            else:
                return "Pear"

        # ── DIAMOND: prominent cheekbones relative to both forehead and jaw ─────
        if c_j > 1.30 and f_j < 1.15:
            return "Diamond"

        # ── HEART vs OVAL: not reliably separable geometrically (see docstring).
        #    Weak tiebreaker on f_j; secondary_hint is always forced for this
        #    branch in analyze_face_shape() below.
        if f_j > 1.21:
            return "Heart"
        return "Oval"

    primary_shape = classify(h_w_ratio, forehead_jaw_ratio, cheek_jaw_ratio)

    leaning_alt = None

    # Check sensitivity margins (boundaries match the thresholds in classify() above)
    if abs(h_w_ratio - 1.225) <= 0.02:
        alt_val = 1.205 if h_w_ratio > 1.225 else 1.245
        alt = classify(alt_val, forehead_jaw_ratio, cheek_jaw_ratio)
        if alt != primary_shape:
            leaning_alt = alt
    elif abs(h_w_ratio - 1.175) <= 0.02:
        alt_val = 1.155 if h_w_ratio > 1.175 else 1.195
        alt = classify(alt_val, forehead_jaw_ratio, cheek_jaw_ratio)
        if alt != primary_shape:
            leaning_alt = alt
    elif abs(forehead_jaw_ratio - 0.92) <= 0.02:
        alt_val = 0.90 if forehead_jaw_ratio > 0.92 else 0.94
        alt = classify(h_w_ratio, alt_val, cheek_jaw_ratio)
        if alt != primary_shape:
            leaning_alt = alt
    elif abs(forehead_jaw_ratio - 1.17) <= 0.02:
        alt_val = 1.15 if forehead_jaw_ratio > 1.17 else 1.19
        alt = classify(h_w_ratio, alt_val, cheek_jaw_ratio)
        if alt != primary_shape:
            leaning_alt = alt
    elif abs(cheek_jaw_ratio - 1.24) <= 0.02:
        alt_val = 1.22 if cheek_jaw_ratio > 1.24 else 1.26
        alt = classify(h_w_ratio, forehead_jaw_ratio, alt_val)
        if alt != primary_shape:
            leaning_alt = alt
    elif abs(cheek_jaw_ratio - 1.30) <= 0.02:
        alt_val = 1.28 if cheek_jaw_ratio > 1.30 else 1.32
        alt = classify(h_w_ratio, forehead_jaw_ratio, alt_val)
        if alt != primary_shape:
            leaning_alt = alt

    # The Heart/Oval boundary (f_j vs 1.21) carries no reliable geometric signal
    # (validated against a labeled dataset: Cohen's d < 0.22 on every ratio we
    # tested). Whenever classify() lands on Heart or Oval, always surface the
    # other as a secondary hint rather than presenting a falsely confident
    # single label.
    if primary_shape in ("Heart", "Oval") and leaning_alt is None:
        leaning_alt = "Oval" if primary_shape == "Heart" else "Heart"

    return {
        "primary_shape": primary_shape,
        "secondary_hint": leaning_alt,
        "measurements": {
            "H/W Ratio": round(h_w_ratio, 2),
            "F/J Ratio": round(forehead_jaw_ratio, 2),
            "C/J Ratio": round(cheek_jaw_ratio, 2)
        }
    }

def cv2_to_std_lab(l_cv2, a_cv2, b_cv2):

    l_std = (l_cv2 * 100.0) / 255.0
    a_std = a_cv2 - 128.0
    b_std = b_cv2 - 128.0
    return l_std, a_std, b_std

def analyze_color(image, landmarks, wrist_image=None):
    h, w, _ = image.shape

    rois = []

    for idx in [10, 9, 8, 151, 65, 295, 330, 101, 50, 280]: 
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        y1, y2 = max(0, cy-10), min(h, cy+10)
        x1, x2 = max(0, cx-10), min(w, cx+10)
        patch = image[y1:y2, x1:x2]

        if patch.size > 0:
            avg_b = np.median(patch[:, :, 0])
            avg_g = np.median(patch[:, :, 1])
            avg_r = np.median(patch[:, :, 2])
            rois.append(np.array([avg_b, avg_g, avg_r]))

    if not rois: return "#FFFFFF", "Neutral", (100.0, 0, 0), "Spring", "Type I"

    rois = np.array(rois)

    luminance = 0.299 * rois[:, 2] + 0.587 * rois[:, 1] + 0.114 * rois[:, 0]

    threshold = np.percentile(luminance, 50)
    well_lit_rois = rois[luminance >= threshold]

    dominant_color = np.median(well_lit_rois, axis=0) 
    dominant_rgb = dominant_color[::-1].astype(int)
    hex_code = '#{:02x}{:02x}{:02x}'.format(*dominant_rgb)

    lab_color_cv2 = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2LAB)[0][0]

    l_raw = float(lab_color_cv2[0])
    a_raw = float(lab_color_cv2[1])
    b_raw = float(lab_color_cv2[2])

    l_std, a_std, b_std = cv2_to_std_lab(l_raw, a_raw, b_raw)

    diffs_mst = MST_X - np.array([l_std, a_std, b_std], dtype=np.float32)
    dists_mst = np.sqrt(np.sum(diffs_mst * diffs_mst, axis=1))
    mst_level = int(np.argmin(dists_mst))
    mst_label = get_mst_label(mst_level)

    diffs_season = SEASON_X - np.array([l_std, a_std, b_std], dtype=np.float32)
    dists_season = np.sqrt(np.sum(diffs_season * diffs_season, axis=1))
    k = min(3, len(SEASON_Y))
    nn_idx = np.argsort(dists_season)[:k]
    votes = {}
    for i in nn_idx:
        lab = SEASON_Y[int(i)]
        votes[lab] = votes.get(lab, 0) + 1
    season_12 = max(votes, key=votes.get) if votes else None

    undertone_found = False

    if wrist_image is not None:
        wh, ww, _ = wrist_image.shape

        cy, cx = wh // 2, ww // 2
        ry, rx = int(wh * 0.25), int(ww * 0.25)
        wrist_patch = wrist_image[max(0, cy-ry):min(wh, cy+ry), max(0, cx-rx):min(ww, cx+rx)]

        if wrist_patch.size > 0:

            wb_lab = cv2.cvtColor(wrist_patch, cv2.COLOR_BGR2LAB)
            avg_a = np.average(wb_lab[:, :, 1])
            avg_b = np.average(wb_lab[:, :, 2])
            wb_lab[:, :, 1] = wb_lab[:, :, 1] - ((avg_a - 128) * (wb_lab[:, :, 0] / 255.0) * 1.5)
            wb_lab[:, :, 2] = wb_lab[:, :, 2] - ((avg_b - 128) * (wb_lab[:, :, 0] / 255.0) * 1.5)
            wb_patch = cv2.cvtColor(wb_lab, cv2.COLOR_LAB2BGR)

            lab = cv2.cvtColor(wb_patch, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0]
            skin_mask = (L > 20) & (L < 240)

            if np.sum(skin_mask) > 0:

                G = wb_patch[:, :, 1]

                valid_G = G[skin_mask]

                g_thresh = np.percentile(valid_G, 25) 

                vein_mask = skin_mask & (G < g_thresh)
                vein_bgr_pixels = wb_patch[vein_mask]

                if vein_bgr_pixels.size > 0:
                    avg_b = np.mean(vein_bgr_pixels[:, 0])
                    avg_g = np.mean(vein_bgr_pixels[:, 1])
                    avg_r = np.mean(vein_bgr_pixels[:, 2])

                    raw_bgr_pixels = wrist_patch[vein_mask]
                    raw_b = np.mean(raw_bgr_pixels[:, 0])
                    raw_g = np.mean(raw_bgr_pixels[:, 1])

                    diff = abs(avg_b - avg_g)
                    confidence = diff / (avg_b + avg_g + 1e-6)

                    neutral_threshold = 0.035 

                    undertone_base = "Warm"
                    if confidence > neutral_threshold or avg_r > avg_b:
                        undertone_base = "Neutral" 
                    elif avg_b > avg_g:
                        undertone_base = "Cool" 
                    else:
                        undertone_base = "Warm"

                    leaning = ""
                    if undertone_base == "Neutral":
                        if raw_g > raw_b:
                            leaning = " (Warm Leaning)"
                        elif raw_b > raw_g:
                            leaning = " (Cool Leaning)"
                    elif undertone_base == "Cool":
                        if raw_g > raw_b:  
                            leaning = " (Neutral Leaning)"
                    elif undertone_base == "Warm":
                        if raw_b > raw_g:  
                            leaning = " (Neutral Leaning)"

                    undertone = f"{undertone_base}{leaning}"        
                    undertone_found = True

    if not undertone_found:
        diff = abs(a_std - b_std)
        if diff < 5:
            undertone = "Neutral"
        elif b_std > a_std:
            undertone = "Warm"
        else:
            undertone = "Cool"

    return hex_code, undertone, (l_std, a_std, b_std), season_12, mst_label

def classify_skin_tone(l_std):

    if l_std > 65:
        return "Light"
    elif l_std >= 45:
        return "Medium"
    else:
        return "Dark"

def analyze_contrast(image, landmarks):

    h, w, _ = image.shape

    skin_lums = []
    for idx in [10, 9, 8, 151, 50, 280]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        y1, y2 = max(0, cy - 8), min(h, cy + 8)
        x1, x2 = max(0, cx - 8), min(w, cx + 8)
        patch = image[y1:y2, x1:x2]
        if patch.size > 0:
            lum = (0.299 * float(np.median(patch[:, :, 2]))
                   + 0.587 * float(np.median(patch[:, :, 1]))
                   + 0.114 * float(np.median(patch[:, :, 0])))
            skin_lums.append(lum)

    feature_lums = []
    for idx in [70, 63, 105, 66, 336, 296, 334, 293, 13, 14, 78, 308]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        y1, y2 = max(0, cy - 5), min(h, cy + 5)
        x1, x2 = max(0, cx - 5), min(w, cx + 5)
        patch = image[y1:y2, x1:x2]
        if patch.size > 0:
            lum = (0.299 * float(np.median(patch[:, :, 2]))
                   + 0.587 * float(np.median(patch[:, :, 1]))
                   + 0.114 * float(np.median(patch[:, :, 0])))
            feature_lums.append(lum)


    if not skin_lums or not feature_lums:
        return "Medium"
    delta = float(np.mean(skin_lums)) - float(np.mean(feature_lums))
    if delta > 80:
        return "High"
    elif delta > 35:
        return "Medium"
    else:
        return "Low"

def get_fashion_palette(skin_tone, undertone_base, contrast):

    if undertone_base == "Warm":
        rec = [
            {"name": "Olive Green",      "hex": "#556B2F"},
            {"name": "Mustard Yellow",   "hex": "#E1AD01"},
            {"name": "Rust",             "hex": "#B7410E"},
            {"name": "Terracotta",       "hex": "#E2725B"},
            {"name": "Coral",            "hex": "#FF7F50"},
            {"name": "Peach",            "hex": "#FFDAB9"},
            {"name": "Tomato Red",       "hex": "#FF6347"},
            {"name": "Maroon",           "hex": "#800000"},
            {"name": "Camel",            "hex": "#C19A6B"},
            {"name": "Chocolate Brown",  "hex": "#D2691E"},
        ]
        acc = [
            {"name": "Gold",         "hex": "#D4AF37"},
            {"name": "Burnt Orange", "hex": "#CC5500"},
            {"name": "Amber",        "hex": "#FFBF00"},
            {"name": "Warm Teal",    "hex": "#008080"},
        ]
        avoid = [
            {"name": "Icy Blue",     "hex": "#F0F8FF"},
            {"name": "Lavender",     "hex": "#E6E6FA"},
            {"name": "Cool Grey",    "hex": "#808080"},
            {"name": "Neon Purple",  "hex": "#BC13FE"},
        ]
        neutrals = [
            {"name": "Ivory",        "hex": "#FFFFF0"},
            {"name": "Warm Beige",   "hex": "#F5F5DC"},
            {"name": "Khaki",        "hex": "#C3B091"},
            {"name": "Warm Taupe",   "hex": "#B38B6D"},
            {"name": "Oatmeal",      "hex": "#EAE0C8"},
        ]
        why = "Warm undertone. Earthy, amber, and terracotta tones work best. Avoid icy blues and lavenders."

    elif undertone_base == "Cool":
        rec = [
            {"name": "Navy Blue",     "hex": "#000080"},
            {"name": "Royal Blue",    "hex": "#4169E1"},
            {"name": "Emerald Green", "hex": "#50C878"},
            {"name": "Sapphire",      "hex": "#0F52BA"},
            {"name": "Plum",          "hex": "#8E4585"},
            {"name": "Burgundy",      "hex": "#800020"},
            {"name": "Rose Pink",     "hex": "#FF66CC"},
            {"name": "Fuchsia",       "hex": "#FF00FF"},
            {"name": "Soft Lavender", "hex": "#D8BFD8"},
            {"name": "Charcoal Grey", "hex": "#36454F"},
        ]
        acc = [
            {"name": "Silver",       "hex": "#C0C0C0"},
            {"name": "Icy Blue",     "hex": "#F0F8FF"},
            {"name": "Magenta",      "hex": "#FF00FF"},
            {"name": "Deep Violet",  "hex": "#330066"},
        ]
        avoid = [
            {"name": "Mustard",      "hex": "#FFDB58"},
            {"name": "Orange",       "hex": "#FFA500"},
            {"name": "Yellow-Green", "hex": "#9ACD32"},
            {"name": "Warm Beige",   "hex": "#F5F5DC"},
        ]
        neutrals = [
            {"name": "Pure White",   "hex": "#FFFFFF"},
            {"name": "Charcoal",     "hex": "#36454F"},
            {"name": "Cool Taupe",   "hex": "#D2B48C"},
            {"name": "Medium Grey",  "hex": "#BEBEBE"},
            {"name": "Slate",        "hex": "#708090"},
        ]
        why = "Cool undertone. Jewel tones and blue-based shades work best. Avoid mustard and warm oranges."

    else:  
        rec = [
            {"name": "Teal",         "hex": "#008080"},
            {"name": "Dusty Pink",   "hex": "#DCAE96"},
            {"name": "Soft Peach",   "hex": "#FFDAB9"},
            {"name": "Jade Green",   "hex": "#00A86B"},
            {"name": "Medium Grey",  "hex": "#BEBEBE"},
            {"name": "Navy",         "hex": "#000080"},
            {"name": "Off-White",    "hex": "#FAF9F6"},
            {"name": "Mauve",        "hex": "#E0B0FF"},
            {"name": "Slate Blue",   "hex": "#6A5ACD"},
            {"name": "Dusty Sage",   "hex": "#A2AD91"},
        ]
        acc = [
            {"name": "Rose Gold",    "hex": "#B76E79"},
            {"name": "Soft Coral",   "hex": "#FF8B8B"},
            {"name": "Aqua",         "hex": "#00FFFF"},
            {"name": "Warm Lilac",   "hex": "#D4B2D8"},
        ]
        avoid = [
            {"name": "Neon Yellow",  "hex": "#FFFF00"},
            {"name": "Neon Green",   "hex": "#39FF14"},
            {"name": "Acid Orange",  "hex": "#FF8F00"},
            {"name": "Dull Mud Brown","hex": "#70543E"},
        ]
        neutrals = [
            {"name": "Stone",        "hex": "#8B8680"},
            {"name": "Off-White",    "hex": "#FAF9F6"},
            {"name": "Warm Sand",    "hex": "#C2B280"},
            {"name": "Cool Linen",   "hex": "#E9DCC9"},
            {"name": "Medium Taupe", "hex": "#674C47"},
        ]
        why = "Neutral undertone. Both warm and cool tones work. Avoid neons and very dull browns."

    if skin_tone == "Light":
        avoid.append({"name": "Pale Yellow", "hex": "#FFFFE0"})
        avoid.append({"name": "Baby Pink",   "hex": "#F4C2C2"})
    elif skin_tone == "Dark":
        rec.append({"name": "Electric Blue", "hex": "#7DF9FF"})
        rec.append({"name": "Bright White",  "hex": "#FFFFFF"})

    if contrast == "High":
        bold_add = [
            {"name": "Jet Black",   "hex": "#0A0A0A"},
            {"name": "Pure White",  "hex": "#FFFFFF"},
            {"name": "Bold Red",    "hex": "#FF0000"},
        ]

        existing_names = {c["name"] for c in bold_add}
        rec = bold_add + [c for c in rec if c["name"] not in existing_names]
        avoid += [
            {"name": "Dusty Rose",  "hex": "#DCAE96"},
            {"name": "Sage Mist",   "hex": "#B2AC88"},
        ]
    elif contrast == "Low":
        soft_add = [
            {"name": "Soft Pink",    "hex": "#FFB6C1"},
            {"name": "Sage Green",   "hex": "#8A9A5B"},
            {"name": "Powder Blue",  "hex": "#B0E0E6"},
            {"name": "Dusty Mauve",  "hex": "#997070"},
        ]
        rec = soft_add + rec[:6]
        avoid += [
            {"name": "Jet Black",    "hex": "#0A0A0A"},
            {"name": "Bright White", "hex": "#FFFFFF"},
        ]

    return {
        "recommended": rec[:12],
        "accents":     acc[:6],
        "avoid":       avoid[:6],
        "neutrals":    neutrals[:5],
        "why":         why,
    }
def draw_landmarks(image, landmarks):
    h, w, _ = image.shape

    for landmark in landmarks:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)
    return image

def swatch_face(image, landmarks, foundation_hex, concealer_hex):
    h, w, _ = image.shape

    l_cheek = landmarks[234]
    r_cheek = landmarks[454]
    face_width = np.sqrt(((l_cheek.x - r_cheek.x) * w)**2 + ((l_cheek.y - r_cheek.y) * h)**2)

    f_radius = max(5, int(face_width * 0.09))
    c_radius = max(4, int(face_width * 0.075))

    cheek_idx = 234
    cheek_pt = (int(landmarks[cheek_idx].x * w), int(landmarks[cheek_idx].y * h))

    eye_idx = 101 
    eye_pt = (int(landmarks[eye_idx].x * w), int(landmarks[eye_idx].y * h))

    f_rgb = hex_to_rgb(foundation_hex)[::-1] 
    c_rgb = hex_to_rgb(concealer_hex)[::-1] 

    cv2.circle(image, cheek_pt, f_radius, f_rgb, -1)
    cv2.circle(image, cheek_pt, f_radius + 2, (255, 255, 255), 2)
    cv2.putText(image, "Foundation", (cheek_pt[0]-f_radius-15, cheek_pt[1]+f_radius+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.circle(image, eye_pt, c_radius, c_rgb, -1)
    cv2.circle(image, eye_pt, c_radius + 2, (255, 255, 255), 2)
    cv2.putText(image, "Concealer", (eye_pt[0]-c_radius-10, eye_pt[1]+c_radius+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image

PALETTES = {
    "Spring": {"colors": ["#FFD700", "#FFA500", "#FF7F50", "#98FB98"]},
    "Summer": {"colors": ["#ADD8E6", "#E6E6FA", "#F08080", "#D3D3D3"]},
    "Autumn": {"colors": ["#8B4513", "#CD853F", "#D2691E", "#A0522D"]},
    "Winter": {"colors": ["#000080", "#800080", "#000000", "#FFFFFF"]},

    "Light Spring": {"colors": ["#FFFACD", "#FFEFD5", "#F0E68C", "#E0FFFF"]},
    "True Spring": {"colors": ["#FFD700", "#FF8C00", "#FF4500", "#32CD32"]},
    "Bright Spring": {"colors": ["#FFFF00", "#FF00FF", "#00FFFF", "#7FFF00"]},
    "Light Summer": {"colors": ["#E0FFFF", "#B0E0E6", "#F0F8FF", "#FFF0F5"]},
    "True Summer": {"colors": ["#0000FF", "#8A2BE2", "#5F9EA0", "#4682B4"]},
    "Soft Summer": {"colors": ["#778899", "#B0C4DE", "#BC8F8F", "#A9A9A9"]},
    "Soft Autumn": {"colors": ["#DEB887", "#D2B48C", "#BC8F8F", "#8FBC8F"]},
    "True Autumn": {"colors": ["#8B4513", "#A52A2A", "#D2691E", "#556B2F"]},
    "Warm Autumn": {"colors": ["#D2691E", "#B8860B", "#CD853F", "#8B0000"]},
    "Deep Winter": {"colors": ["#191970", "#4B0082", "#2F4F4F", "#000000"]},
    "True Winter": {"colors": ["#000080", "#800080", "#008080", "#C0C0C0"]},
    "Bright Winter": {"colors": ["#0000FF", "#FF00FF", "#00FF00", "#FFFFFF"]},
}

with app.app_context():
    db.create_all()

    if Foundation.query.count() == 0:
        print("Database empty. Auto-syncing shades...")
        import sync_shades
        sync_shades.sync()

@app.route('/sw.js')
def serve_sw():
    return app.send_static_file('sw.js')

@app.route('/manifest.json')
def serve_manifest():
    return app.send_static_file('manifest.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if request.method == 'POST':
        identifier = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['last_active'] = datetime.now().timestamp()
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('auth/login.html')

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def signup():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        gender = request.form.get('gender')
        password = request.form.get('password')

        username = email.split('@')[0] if email and '@' in email else email

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():

            username = email

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, email=email, full_name=full_name, gender=gender, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        session['last_active'] = datetime.now().timestamp()
        return redirect(url_for('dashboard'))
    return render_template('auth/signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('last_active', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('features/dashboard.html', user=current_user)

@app.route('/chroma-skin', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def chroma_skin():
    skin_hex = None
    undertone = None
    error = None
    if request.method == 'GET' and current_user.is_authenticated and current_user.skin_hex:
        return render_template('features/chroma_skin.html', 
                               skin_hex=current_user.skin_hex, 
                               undertone=current_user.undertone, 
                               mst_label=current_user.mst_label, 
                               season=current_user.season)

    if request.method == 'POST':
        if not current_user.is_authenticated:
            if session.get('guest_analyses_count', 0) >= 1:
                return render_template('features/chroma_skin.html', error="Guest limit reached. Please register to perform more analyses and save results.")

        if 'image' in request.files:
            file = request.files['image']
            wrist_file = request.files.get('wrist_image')

            if file.filename != '' and allowed_file(file.filename):
                filename = secure_upload_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                if not validate_image_file(filepath):
                    error = "invalid_file"
                else:
                    image = cv2.imread(filepath)
                    if image is None:
                        error = "invalid_file"
                    else:
                        wrist_image = None
                        if wrist_file and wrist_file.filename != '':
                            wrist_filename = secure_upload_filename(wrist_file.filename)
                            wrist_filepath = os.path.join(UPLOAD_FOLDER, wrist_filename)
                            wrist_file.save(wrist_filepath)
                            if validate_image_file(wrist_filepath):
                                wrist_image = cv2.imread(wrist_filepath)

                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        detection_result = detector.detect(mp_image)

                        if detection_result.face_landmarks:
                            skin_hex, undertone, lab_std, season_12, mst_label = analyze_color(image, detection_result.face_landmarks[0], wrist_image=wrist_image)

                            if current_user.is_authenticated:
                                current_user.skin_hex = skin_hex
                                current_user.undertone = undertone
                                current_user.season = season_12
                                current_user.mst_label = mst_label
                                db.session.commit()
                            else:
                                session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1

                            return render_template('features/chroma_skin.html', skin_hex=skin_hex, undertone=undertone, uploaded_image=filename, mst_label=mst_label, season=season_12)
                        else:
                            error = "no_face_detected"
            else:
                error = "no_file"

    return render_template('features/chroma_skin.html', error=error)

@app.route('/smart-match', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def smart_match():

    brands = db.session.query(Foundation.brand).distinct().all()
    brands = sorted([b[0] for b in brands if b[0]])

    matches = None
    concealer_matches = None
    skin_hex = None
    swatched_image = None
    error = None

    if request.method == 'GET' and current_user.is_authenticated and current_user.skin_hex:
        skin_hex = current_user.skin_hex
        selected_brand = request.args.get('brand') or request.form.get('brand')
        try:
            l_user, a_user, b_user = hex_to_lab(skin_hex)
            
            brand_has_no_foundations = False
            brand_has_no_concealers = False
            if selected_brand:
                f_count = Foundation.query.filter_by(brand=selected_brand, category='Foundation').count()
                c_count = Foundation.query.filter_by(brand=selected_brand, category='Concealer').count()
                brand_has_no_foundations = (f_count == 0)
                brand_has_no_concealers = (c_count == 0)

            f_query = Foundation.query.filter_by(category='Foundation')
            if selected_brand:
                f_query = f_query.filter_by(brand=selected_brand)

            foundations = f_query.all()
            if foundations:
                f_labs = np.array([[f.l, f.a, f.b] for f in foundations])
                distances = delta_e_cie2000_vec([l_user, a_user, b_user], f_labs)

                top_indices = np.argsort(distances)
                matches = []
                for i in top_indices:
                    dist = distances[i]
                    if dist > 25.0: continue 
                    f = foundations[i]
                    f.match_score = max(1, min(99, int(100 - (dist * 3.5))))
                    matches.append(f)
                    if len(matches) >= 3: break

            target_l = min(100.0, l_user + 5.0)
            c_query = Foundation.query.filter_by(category='Concealer')
            if selected_brand:
                c_query = c_query.filter_by(brand=selected_brand)
            concealers = c_query.all()

            if concealers:
                c_labs = np.array([[c.l, c.a, c.b] for c in concealers])
                c_dists = delta_e_cie2000_vec([target_l, a_user, b_user], c_labs)

                top_c_indices = np.argsort(c_dists)
                concealer_matches = []
                for i in top_c_indices:
                    dist = c_dists[i]
                    if dist > 25.0: continue 
                    c = concealers[i]
                    c.match_score = max(1, min(99, int(100 - (dist * 3.5))))
                    concealer_matches.append(c)
                    if len(concealer_matches) >= 3: break
            
            return render_template('features/smart_match.html', 
                                 brands=brands, selected_brand=selected_brand,
                                 matches=matches, concealer_matches=concealer_matches, 
                                 skin_hex=skin_hex, swatched_image=None, 
                                 swatch_coords=None, season=current_user.season, mst_label=current_user.mst_label,
                                 brand_has_no_foundations=brand_has_no_foundations,
                                 brand_has_no_concealers=brand_has_no_concealers)
        except Exception as e:
            error = f"Error preloading matches: {e}"

    if request.method == 'POST':
        if not current_user.is_authenticated:
            if session.get('guest_analyses_count', 0) >= 1:
                return render_template('features/smart_match.html', brands=brands, error="Guest limit reached.")

        file = request.files.get('image')
        existing_filename = request.form.get('existing_image')
        selected_brand = request.form.get('brand')

        filename = None
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_upload_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            if not validate_image_file(filepath):
                error = "invalid_file"
                filename = None
        elif file and file.filename != '':
            error = "invalid_file_type"
        elif existing_filename:
            filename = existing_filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)

        if not filename:
            error = "no_file"

        if filename:
            image = cv2.imread(filepath)
            if image is None:
                error = "no_file"
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)

                if detection_result.face_landmarks:
                    hex_code, undertone, lab_std, season_12, mst_label = analyze_color(image, detection_result.face_landmarks[0])
                    skin_hex = hex_code
                    l_user, a_user, b_user = lab_std

                    if current_user.is_authenticated:
                        current_user.skin_hex = skin_hex
                        current_user.undertone = undertone
                        current_user.season = season_12
                        current_user.mst_label = mst_label
                        db.session.commit()
                    else:
                        session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1

                    landmarks = detection_result.face_landmarks[0]
                    f_swatch_idx, c_swatch_idx = 58, 101
                    swatch_coords = {
                        'foundation': {'x': landmarks[f_swatch_idx].x, 'y': landmarks[f_swatch_idx].y},
                        'concealer': {'x': landmarks[c_swatch_idx].x, 'y': landmarks[c_swatch_idx].y}
                    }

                    swatched_image = 'clean_' + filename
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, swatched_image), image)
                else:
                    error = "no_face_detected"

            if not error:

                brand_has_no_foundations = False
                brand_has_no_concealers = False
                if selected_brand:
                    f_count = Foundation.query.filter_by(brand=selected_brand, category='Foundation').count()
                    c_count = Foundation.query.filter_by(brand=selected_brand, category='Concealer').count()
                    brand_has_no_foundations = (f_count == 0)
                    brand_has_no_concealers = (c_count == 0)

                f_query = Foundation.query.filter_by(category='Foundation')
                if selected_brand:
                    f_query = f_query.filter_by(brand=selected_brand)

                foundations = f_query.all()
                if foundations:
                    f_labs = np.array([[f.l, f.a, f.b] for f in foundations])
                    distances = delta_e_cie2000_vec([l_user, a_user, b_user], f_labs)

                    top_indices = np.argsort(distances)
                    matches = []
                    for i in top_indices:
                        dist = distances[i]
                        if dist > 25.0: continue 
                        f = foundations[i]
                        f.match_score = max(1, min(99, int(100 - (dist * 3.5))))
                        matches.append(f)
                        if len(matches) >= 3: break

                target_l = min(100.0, l_user + 5.0)
                c_query = Foundation.query.filter_by(category='Concealer')
                if selected_brand:
                    c_query = c_query.filter_by(brand=selected_brand)
                concealers = c_query.all()

                if concealers:
                    c_labs = np.array([[c.l, c.a, c.b] for c in concealers])
                    c_dists = delta_e_cie2000_vec([target_l, a_user, b_user], c_labs)

                    top_c_indices = np.argsort(c_dists)
                    concealer_matches = []
                    for i in top_c_indices:
                        dist = c_dists[i]
                        if dist > 25.0: continue 
                        c = concealers[i]
                        c.match_score = max(1, min(99, int(100 - (dist * 3.5))))
                        concealer_matches.append(c)
                        if len(concealer_matches) >= 3: break

                return render_template('features/smart_match.html', 
                                     brands=brands, selected_brand=selected_brand,
                                     matches=matches, concealer_matches=concealer_matches, 
                                     skin_hex=skin_hex, swatched_image=swatched_image, 
                                     swatch_coords=swatch_coords, season=season_12, mst_label=mst_label,
                                     brand_has_no_foundations=brand_has_no_foundations,
                                     brand_has_no_concealers=brand_has_no_concealers)

    return render_template('features/smart_match.html', brands=brands, matches=matches, error=error)

@app.route('/morpho-face')
def morpho_face():

    return render_template('features/morpho_face.html', shape="Unknown", description="")

@app.route('/vibe-check', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def vibe_check():
    season         = None
    fashion_palette = None
    skin_tone      = None
    undertone_base = None
    contrast       = None
    uploaded_image = None
    error          = None
    mst_label      = None

    if request.method == 'GET' and current_user.is_authenticated:
        palette_entry = UserPalette.query.get(current_user.id)
        if palette_entry and palette_entry.palette_json:
            try:
                palette_data = json.loads(palette_entry.palette_json)
                if "palette" in palette_data:
                    fashion_palette = palette_data["palette"]
                    skin_tone = palette_data.get("skin_tone")
                    undertone_base = palette_data.get("undertone_base")
                    contrast = palette_data.get("contrast")
                else:
                    fashion_palette = palette_data
                    if current_user.skin_hex:
                        l_user, _, _ = hex_to_lab(current_user.skin_hex)
                        skin_tone = classify_skin_tone(l_user)
                    if current_user.undertone:
                        undertone_base = current_user.undertone.split(" (")[0]
                    contrast = "Medium"
                season = current_user.season
                mst_label = current_user.mst_label
            except Exception as e:
                print(f"Error loading UserPalette on GET: {e}")

    if request.method == 'POST':
        if not current_user.is_authenticated:
            if session.get('guest_analyses_count', 0) >= 1:
                return render_template('features/vibe_check.html',
                                       error="Guest limit reached. Please register to perform more analyses and save results.")

        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_upload_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                if not validate_image_file(filepath):
                    error = "invalid_file"
                else:
                    image = cv2.imread(filepath)
                    if image is None:
                        error = "invalid_file"
                    else:
                        uploaded_image = filename
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    detection_result = detector.detect(mp_image)

                    if detection_result.face_landmarks:
                        landmarks = detection_result.face_landmarks[0]

                        skin_hex, undertone, lab_std, season, mst_label = analyze_color(image, landmarks)

                        skin_tone      = classify_skin_tone(lab_std[0])
                        contrast       = analyze_contrast(image, landmarks)
                        undertone_base = undertone.split(" (")[0]   

                        fashion_palette = get_fashion_palette(skin_tone, undertone_base, contrast)

                        if current_user.is_authenticated:
                            current_user.undertone  = undertone
                            current_user.skin_hex   = skin_hex
                            current_user.season     = season
                            current_user.mst_label  = mst_label
                            # Save palette for wardrobe integration
                            palette_data = {
                                "skin_tone": skin_tone,
                                "undertone_base": undertone_base,
                                "contrast": contrast,
                                "palette": fashion_palette
                            }
                            existing_palette = UserPalette.query.get(current_user.id)
                            if existing_palette:
                                existing_palette.palette_json = json.dumps(palette_data)
                                existing_palette.updated_at = datetime.utcnow()
                            else:
                                new_palette = UserPalette(
                                    user_id=current_user.id,
                                    palette_json=json.dumps(palette_data)
                                )
                                db.session.add(new_palette)
                            db.session.commit()
                        else:
                            session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1
                    else:
                        error = "no_face_detected"
            elif file.filename != '':
                error = "invalid_file_type"
            else:
                error = "no_file"
        else:
            error = "no_file"

    return render_template('features/vibe_check.html',
                           season=season,
                           fashion_palette=fashion_palette,
                           skin_tone=skin_tone,
                           undertone_base=undertone_base,
                           contrast=contrast,
                           uploaded_image=uploaded_image,
                           error=error,
                           mst_label=mst_label)
@app.route('/morpho-analyze', methods=['POST'])
@limiter.limit("10 per minute")
def morpho_analyze():
    if not current_user.is_authenticated:
        if session.get('guest_analyses_count', 0) >= 1:
            return jsonify({'error': 'Guest limit reached. Please register to perform more analyses and save results.'}), 403

    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No image selected'}), 400

    filename = secure_upload_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    if not validate_image_file(filepath):
        return jsonify({'error': 'Invalid image file'}), 400
    image = cv2.imread(filepath)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks[0]

        matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None
        height, width, _ = image.shape
        face_shape_data = analyze_face_shape(face_landmarks_list, width, height, matrix)
        face_shape = face_shape_data['primary_shape']

        if current_user.is_authenticated:
            current_user.face_shape = face_shape
            db.session.commit()
        else:
            session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1

        image = draw_landmarks(image, face_landmarks_list)
        output_path = os.path.join(UPLOAD_FOLDER, 'morpho_' + filename)
        cv2.imwrite(output_path, image)

        return jsonify({
            'shape': face_shape,
            'secondary_hint': face_shape_data.get('secondary_hint'),
            'measurements': face_shape_data.get('measurements'),
            'description': RECOMMENDATIONS.get(face_shape.split(" (")[0]),
            'image_url': url_for('static', filename='uploads/morpho_' + filename)
        })

    return jsonify({'error': 'No face detected'}), 400

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w, _ = background.shape

    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape

    # Calculate overlap region in background coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    # If no overlap, return background unchanged
    if x1 >= x2 or y1 >= y2:
        return background

    # Calculate corresponding overlap region in overlay coordinates
    ol_x1 = max(0, -x)
    ol_y1 = max(0, -y)
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)

    # Slice the images to the overlap region
    overlay_crop = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    background_roi = background[y1:y2, x1:x2].astype(float)

    has_alpha = overlay_crop.shape[2] == 4

    if not has_alpha:
        if overlay_crop.shape[2] == 3:
            overlay_crop = cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(overlay_crop, cv2.COLOR_BGRA2GRAY)
        _, mask_bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        mask_bw = cv2.erode(mask_bw, kernel, iterations=1)
        overlay_crop = overlay_crop.copy()
        overlay_crop[:, :, 3] = mask_bw

    overlay_img = overlay_crop[:, :, :3].astype(float)
    mask = overlay_crop[:, :, 3] / 255.0
    mask = cv2.merge([mask, mask, mask])

    blended = (background_roi * (1.0 - mask)) + (overlay_img * mask)
    background[y1:y2, x1:x2] = blended.astype('uint8')

    return background

EYEWEAR_ASSETS = {
    "Thick Black Square":       "Thick Black Square.png",
    "Narrow Rectangle":         "Narrow Rectangle.png",
    "Oversized Square":         "Oversized Square.png",
    "Lens Var (Square)":        "Lens Var (Square).png",
    "Clubmaster":               "Clubmaster.png",
    "Small Round":              "Small Round.png",
    "Thick Acetate Round":      "Thick Acetate Round.png",
    "Thin Metal Round":         "Thin Metal Round.png",
    "Thin Oval":                "Thin Oval.png",
    "Oversized Round":          "Oversized Round.png",
    "Lens Var (Wayfarer)":      "Lens Var (Wayfarer).png",
    "Lens Var (Aviator)":       "Lens Var (Aviator).png",
    "Hexagonal Geometric":      "Hexagonal Geometric.png",
    "Octagonal Geometric":      "Octagonal Geometric.png",
    "Rimless Rectangle":        "Rimless Rectangle.png",
    "Semi-Rimless (Bottom Rim)": "Semi-Rimless (Bottom Rim).png",
    "Semi-Rimless (Top Rim)":   "Semi-Rimless (Top Rim).png",
    "Thin Metal Square":        "Thin Metal Square.png",
    "Minimal Browline":         "Minimal Browline.png",
    "Sharp Cat-Eye":            "Sharp Cat-Eye.png",
    "Soft Cat-Eye":             "Soft Cat-Eye.png",
    "Thick Vintage Cat-Eye":    "Thick Vintage Cat-Eye.png",
    "Blue Light (Transparent)": "Blue Light (Transparent Frame).png",
    "Fashion Color":            "Fashion Color.png",
    "Prescription Style":       "Prescription Style.png",
    "Thick Acetate Oval":       "Thick Acetate Oval.png",
}

SHADES_FOLDER = "static/assets/Copy of shades"

def compute_face_metrics(landmarks, image_w, image_h, matrix=None):

    def pt(idx):
        return np.array([landmarks[idx].x * image_w, landmarks[idx].y * image_h])

    def dist(a, b):
        return float(np.linalg.norm(pt(a) - pt(b)))

    face_width     = dist(234, 454)         

    face_shape = analyze_face_shape(landmarks, image_w, image_h, matrix=matrix)['primary_shape']

    all_x = [landmarks[i].x for i in range(len(landmarks))]
    all_y = [landmarks[i].y for i in range(len(landmarks))]
    bb_w  = (max(all_x) - min(all_x)) * image_w
    bb_h  = (max(all_y) - min(all_y)) * image_h
    bb_area_frac = (bb_w * bb_h) / (image_w * image_h)
    if bb_area_frac > 0.25:
        face_size = "Large"
    elif bb_area_frac < 0.12:
        face_size = "Small"
    else:
        face_size = "Medium"

    nose_w = dist(129, 358)
    nose_ratio = nose_w / max(face_width, 1)
    if nose_ratio > 0.27:
        nose_width = "Wide"
    elif nose_ratio < 0.18:
        nose_width = "Narrow"
    else:
        nose_width = "Medium"

    inter_eye   = dist(133, 362)   
    left_eye_w  = dist(33,  133)
    right_eye_w = dist(362, 263)
    avg_eye_w   = (left_eye_w + right_eye_w) / 2
    spacing_ratio = inter_eye / max(avg_eye_w, 1)
    if spacing_ratio < 1.0:
        eye_spacing = "Close-set"
    elif spacing_ratio > 1.6:
        eye_spacing = "Wide-set"
    else:
        eye_spacing = "Balanced"

    return {
        "face_shape":   face_shape,
        "face_size":    face_size,
        "nose_width":   nose_width,
        "eye_spacing":  eye_spacing,
    }

def select_eyewear_frames(face_shape, face_size, nose_width, eye_spacing, undertone, contrast):

    shape_rec = {
        "Round":     ["Thick Black Square", "Narrow Rectangle", "Oversized Square",
                      "Lens Var (Square)", "Clubmaster"],
        "Square":    ["Small Round", "Thick Acetate Round", "Thin Metal Round",
                      "Thin Oval", "Oversized Round"],
        "Oval":      ["Lens Var (Wayfarer)", "Clubmaster", "Lens Var (Aviator)",
                      "Oversized Square", "Hexagonal Geometric"],
        "Heart":     ["Lens Var (Aviator)", "Rimless Rectangle", "Semi-Rimless (Bottom Rim)",
                      "Thin Metal Square", "Minimal Browline"],
        "Rectangle": ["Oversized Square", "Oversized Round", "Thick Acetate Oval",
                      "Lens Var (Wayfarer)", "Thick Black Square"],
        "Oblong":    ["Oversized Square", "Oversized Round", "Thick Acetate Oval",
                      "Lens Var (Wayfarer)", "Thick Black Square"],
        "Pear":      ["Oversized Square", "Thick Black Square", "Clubmaster",
                      "Lens Var (Wayfarer)", "Oversized Round"],
        "Triangle":  ["Oversized Square", "Thick Black Square", "Clubmaster",
                      "Lens Var (Wayfarer)", "Oversized Round"],
    }
    shape_avoid = {
        "Round":     ["Small Round", "Oversized Round", "Thin Oval"],
        "Square":    ["Thick Black Square", "Narrow Rectangle", "Sharp Cat-Eye"],
        "Oval":      ["Small Round"],
        "Heart":     ["Sharp Cat-Eye", "Semi-Rimless (Top Rim)", "Thick Black Square"],
        "Rectangle": ["Narrow Rectangle", "Rimless Rectangle", "Small Round"],
        "Oblong":    ["Narrow Rectangle", "Rimless Rectangle", "Small Round"],
        "Pear":      ["Narrow Rectangle", "Small Round", "Thin Oval"],
        "Triangle":  ["Narrow Rectangle", "Small Round", "Thin Oval"],
    }

    face_shape_base = face_shape.split(" (")[0]
    base   = list(shape_rec.get(face_shape_base,   shape_rec["Oval"]))
    avoids = list(shape_avoid.get(face_shape_base, []))

    size_boost = {
        "Small":  ["Thin Metal Round", "Thin Oval", "Rimless Rectangle"],
        "Large":  ["Oversized Square", "Thick Black Square", "Oversized Round"],
        "Medium": [],
    }
    boost = size_boost.get(face_size, [])
    for f in boost:
        if f not in base:
            base.insert(1, f)  

    if nose_width == "Wide":
        for f in ["Lens Var (Aviator)", "Thin Metal Round", "Rimless Rectangle"]:
            if f not in base:
                base.append(f)
    elif nose_width == "Narrow":
        for f in ["Thick Acetate Round", "Thick Black Square"]:
            if f not in base:
                base.append(f)

    if eye_spacing == "Close-set":
        for f in ["Thin Metal Round", "Rimless Rectangle"]:
            if f not in base:
                base.append(f)
    elif eye_spacing == "Wide-set":
        for f in ["Clubmaster", "Thick Black Square"]:
            if f not in base:
                base.append(f)

    undertone_b = undertone.split(" (")[0] if undertone else "Neutral"
    if contrast == "High":
        for f in ["Thick Black Square", "Clubmaster", "Oversized Square"]:
            if f not in base:
                base.insert(0, f)
    elif contrast == "Low":
        for f in ["Rimless Rectangle", "Thin Metal Round", "Blue Light (Transparent)"]:
            if f not in base:
                base.append(f)

    WHY = {
        "Thick Black Square":       "Angular frame counterbalances rounded facial curves.",
        "Narrow Rectangle":         "Horizontal lines add perceived width to the face.",
        "Oversized Square":         "Bold geometry creates strong structural contrast.",
        "Lens Var (Square)":        "Structured lens shape adds definition.",
        "Clubmaster":               "Thick brow-line adds top-face weight and definition.",
        "Small Round":              "Soft circular shape softens angular jaw lines.",
        "Thick Acetate Round":      "Curved frame echoes and softens square proportions.",
        "Thin Metal Round":         "Minimal profile suits smaller or close-set features.",
        "Thin Oval":                "Gentle curves reduce jaw sharpness without bulk.",
        "Oversized Round":          "Wide circle widens the visual field on a square face.",
        "Lens Var (Wayfarer)":      "Balanced trapezoidal shape works across proportions.",
        "Lens Var (Aviator)":       "Teardrop profile is bottom-heavy, balancing wide foreheads.",
        "Hexagonal Geometric":      "Geometric facets add visual interest without extremes.",
        "Octagonal Geometric":      "Multi-sided shape adds angular interest to balanced faces.",
        "Rimless Rectangle":        "Invisible frame suits close-set eyes and small faces.",
        "Semi-Rimless (Bottom Rim)": "Bottom weight balances a prominent forehead or heart face.",
        "Semi-Rimless (Top Rim)":   "Top bar frame adds structure to soft facial features.",
        "Thin Metal Square":        "Delicate square frame balances without adding bulk.",
        "Minimal Browline":         "Subtle brow-line provides definition without heaviness.",
        "Sharp Cat-Eye":            "Upward sweep elongates and lifts facial proportions.",
        "Soft Cat-Eye":             "Gentle upswept corners suit balanced facial geometry.",
        "Thick Vintage Cat-Eye":    "Retro frame adds width and lifts cheekbone emphasis.",
        "Blue Light (Transparent)": "Clear frame is low-contrast and suits low-contrast faces.",
        "Fashion Color":            "Tinted frame adds color pop for high-contrast looks.",
        "Prescription Style":       "Classic neutral shape suits medium and oval faces.",
        "Thick Acetate Oval":       "Large oval frame adds perceived width to long faces.",
    }

    recommended = []
    for name in base:
        fname = EYEWEAR_ASSETS.get(name)
        if fname and os.path.exists(os.path.join(SHADES_FOLDER, fname)):
            recommended.append({
                "name":     name,
                "filename": fname,
                "why":      WHY.get(name, "Structurally complementary to your facial measurements."),
            })
        if len(recommended) >= 5:
            break

    avoid_out = []
    for name in avoids:
        if any(r["name"] == name for r in recommended):
            continue
        fname = EYEWEAR_ASSETS.get(name)
        avoid_out.append({"name": name, "filename": fname})
        if len(avoid_out) >= 3:
            break

    return recommended, avoid_out

def apply_glasses_overlay(base_image, landmarks, glasses_path, face_size="Medium"):
    img = base_image.copy()
    h, w, _ = img.shape

    left_eye  = landmarks[33]
    right_eye = landmarks[263]
    lx, ly = int(left_eye.x  * w), int(left_eye.y  * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)

    l_cheek = landmarks[234]
    r_cheek = landmarks[454]
    cx, cy = int(l_cheek.x * w), int(l_cheek.y * h)
    dx, dy = int(r_cheek.x * w), int(r_cheek.y * h)
    face_width = float(np.linalg.norm(np.array([cx, cy]) - np.array([dx, dy])))

    glasses_width = int(face_width * 1.22)

    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses is None:
        return img

    if glasses.shape[2] == 3:
        glasses = cv2.cvtColor(glasses, cv2.COLOR_BGR2BGRA)

    aspect_ratio    = glasses.shape[0] / glasses.shape[1]
    glasses_height  = int(glasses_width * aspect_ratio)

    center_x = (lx + rx) // 2

    vertical_offset = int(glasses_height * 0.05)
    center_y = (ly + ry) // 2 + vertical_offset

    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))

    # Scale the glasses image first to preserve aspect ratio, then rotate
    glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))
    center_rotated = (glasses_width / 2.0, glasses_height / 2.0)
    M = cv2.getRotationMatrix2D(center_rotated, angle, 1.0)
    glasses_rotated = cv2.warpAffine(glasses_resized, M, (glasses_width, glasses_height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))

    top_left_x = center_x - (glasses_width // 2)
    top_left_y = center_y - (glasses_height // 2)

    img = overlay_transparent(img, glasses_rotated, top_left_x, top_left_y)
    return img

def apply_beard_overlay(base_image, landmarks, beard_path):
    img = base_image.copy()
    h, w, _ = img.shape

    l_jaw = landmarks[58]
    r_jaw = landmarks[288]
    lx, ly = int(l_jaw.x * w), int(l_jaw.y * h)
    rx, ry = int(r_jaw.x * w), int(r_jaw.y * h)

    jaw_width = float(np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry])))
    beard_width = int(jaw_width * 1.15)

    beard = cv2.imread(beard_path, cv2.IMREAD_UNCHANGED)
    if beard is None:
        return img

    if beard.shape[2] == 3:
        beard = cv2.cvtColor(beard, cv2.COLOR_BGR2BGRA)

    aspect_ratio = beard.shape[0] / beard.shape[1]
    beard_height = int(beard_width * aspect_ratio)

    chin = landmarks[152]
    cx, cy = int(chin.x * w), int(chin.y * h)

    center_x = cx
    center_y = cy - int(beard_height * 0.4)

    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))

    # Scale the beard image first to preserve aspect ratio, then rotate
    beard_resized = cv2.resize(beard, (beard_width, beard_height))
    center_rotated = (beard_width / 2.0, beard_height / 2.0)
    M = cv2.getRotationMatrix2D(center_rotated, angle, 1.0)
    beard_rotated = cv2.warpAffine(beard_resized, M, (beard_width, beard_height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))

    top_left_x = center_x - (beard_width // 2)
    top_left_y = center_y - (beard_height // 2)

    img = overlay_transparent(img, beard_rotated, top_left_x, top_left_y)
    return img

@app.route('/frame-fit', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def frame_fit():
    face_analysis   = None   
    recommended     = []     
    avoid           = []     
    active_image    = None   
    raw_image       = None   
    error           = None

    if request.method == 'POST':
        if not current_user.is_authenticated:
            if session.get('guest_analyses_count', 0) >= 1:
                return render_template('features/frame_fit.html',
                                       error="Guest limit reached. Please register to perform more analyses and save results.")

        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_upload_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                if not validate_image_file(filepath):
                    error = "invalid_file"
                else:
                    image = cv2.imread(filepath)
                    raw_image = filename

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    detection_result = detector.detect(mp_image)

                    if detection_result.face_landmarks:
                        if not current_user.is_authenticated:
                            session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1

                        landmarks  = detection_result.face_landmarks[0]
                        matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None
                        img_h, img_w, _ = image.shape

                        metrics   = compute_face_metrics(landmarks, img_w, img_h, matrix=matrix)
                        face_shape  = metrics["face_shape"]
                        face_size   = metrics["face_size"]
                        nose_width  = metrics["nose_width"]
                        eye_spacing = metrics["eye_spacing"]

                        skin_hex, undertone, lab_std, season_12, mst_label = analyze_color(image, landmarks)
                        undertone_base = undertone.split(" (")[0]
                        contrast       = analyze_contrast(image, landmarks)

                        face_analysis = {
                            "face_shape":   face_shape,
                            "face_size":    face_size,
                            "nose_width":   nose_width,
                            "eye_spacing":  eye_spacing,
                            "undertone":    undertone_base,
                            "contrast":     contrast,
                        }

                        recommended_raw, avoid = select_eyewear_frames(
                            face_shape, face_size, nose_width,
                            eye_spacing, undertone_base, contrast
                        )

                        recommended = []
                        for i, frame in enumerate(recommended_raw):
                            glasses_path = os.path.join(SHADES_FOLDER, frame["filename"])
                            try_on_img   = apply_glasses_overlay(image, landmarks, glasses_path, face_size)
                            out_name     = f"tryon_{i}_{filename}"
                            out_path     = os.path.join(UPLOAD_FOLDER, out_name)
                            cv2.imwrite(out_path, try_on_img)
                            recommended.append({
                                **frame,
                                "try_on_image": out_name,
                            })

                        if recommended:
                            active_image = recommended[0]["try_on_image"]

                        if current_user.is_authenticated:
                            current_user.face_shape = face_shape
                            db.session.commit()

                    else:
                        error = "no_face_detected"
            else:
                error = "no_file"
        else:
            error = "no_file"

    return render_template('features/frame_fit.html',
                           face_analysis=face_analysis,
                           recommended=recommended,
                           avoid=avoid,
                           active_image=active_image,
                           raw_image=raw_image,
                           error=error)


@app.route('/grooming-blueprint', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def grooming_blueprint():
    processed_image = None
    recommendations = None
    face_shape = "Unknown"
    gender = request.form.get('gender', 'male')
    error = None

    if request.method == 'POST':
        if not current_user.is_authenticated:
            if session.get('guest_analyses_count', 0) >= 1:
                return render_template('features/grooming.html', error="Guest limit reached.")

        file = request.files.get('image')
        existing_filename = request.form.get('existing_image')

        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_upload_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            if not validate_image_file(filepath):
                error = "Invalid image file."
                filename = None
        elif file and file.filename != '':
            error = "invalid_file_type"
        elif existing_filename:
            filename = existing_filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
        else:
            filename = None

        if filename:
            image = cv2.imread(filepath)
            if image is None:
                error = "Invalid image file."
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    h, w, _ = image.shape

                    matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None
                    face_shape_data = analyze_face_shape(landmarks, w, h, matrix)
                    face_shape = face_shape_data['primary_shape']
                    shape_key = face_shape.split(" (")[0] if face_shape else "Oval"
                    recommendations = GROOMING_RECOMMENDATIONS.get(shape_key, GROOMING_RECOMMENDATIONS["Oval"])

                    if current_user.is_authenticated:
                        current_user.face_shape = face_shape
                        db.session.commit()
                    else:
                        session['guest_analyses_count'] = session.get('guest_analyses_count', 0) + 1

                    output_path = os.path.join(UPLOAD_FOLDER, 'groomed_' + filename)
                    cv2.imwrite(output_path, image)
                    processed_image = 'groomed_' + filename
                else:
                    error = "No face detected."

    return render_template('features/grooming.html', 
                           processed_image=processed_image, 
                           recommendations=recommendations,
                           face_shape=face_shape,
                           gender=gender,
                           error=error)

def get_owner_id():
    if current_user.is_authenticated:
        return str(current_user.id)
    if 'guest_id' not in session:
        session['guest_id'] = str(uuid.uuid4())
    return session['guest_id']

@app.route('/wardrobe')
@limiter.limit("30 per minute")
def wardrobe():
    owner_id = get_owner_id()
    if request.headers.get('Accept') == 'application/json':
        items = WardrobeItem.query.filter_by(owner_id=owner_id).all()
        return jsonify([{
            "id": i.id,
            "image_path": i.image_path,
            "category": i.category,
            "label": i.label,
            "color_palette": json.loads(i.palette_json),
            "color_name": i.color_name,
            "accent_rgb": [int(x) for x in i.accent_rgb.split(',')],
            "base_rgb": [int(x) for x in i.base_rgb.split(',')]
        } for i in items])
    return render_template('features/wardrobe.html')

@app.route('/wardrobe/upload', methods=['POST'])
@limiter.limit("10 per minute")
def wardrobe_upload():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist('images')
    owner_id = get_owner_id()

    current_count = WardrobeItem.query.filter_by(owner_id=owner_id).count()
    if current_count + len(files) > MAX_WARDROBE_ITEMS:
        return jsonify({"error": f"Collection limit reached ({MAX_WARDROBE_ITEMS} items max)"}), 400

    processed_items = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        file_content = file.read()
        file.seek(0)
        img_hash = hashlib.md5(file_content).hexdigest()

        existing = WardrobeItem.query.filter_by(owner_id=owner_id, image_hash=img_hash).first()
        if existing:
            continue

        item_id = str(uuid.uuid4())
        filename = secure_filename(f"wardrobe_{item_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        if not validate_image_file(filepath):
            continue

        try:

            try:
                category, label = wardrobe_logic.classify_item(filepath)
            except Exception as e:
                print(f"AI classification skipped: {e}")
                category, label = "unknown", "item"

            palette = wardrobe_logic.extract_palette(filepath)
            accent, base = wardrobe_logic.pick_accent_and_base(palette)
            color_name = wardrobe_logic.nearest_color_name(base)

            new_item = WardrobeItem(
                id=item_id,
                owner_id=owner_id,
                image_path=url_for('static', filename=f'uploads/{filename}'),
                category=category,
                label=label,
                palette_json=json.dumps(palette),
                accent_rgb=",".join(map(str, accent)),
                base_rgb=",".join(map(str, base)),
                color_name=color_name,
                image_hash=img_hash
            )
            db.session.add(new_item)
            db.session.commit()

            processed_items.append({
                "id": item_id,
                "image_path": new_item.image_path,
                "category": category,
                "label": label,
                "color_palette": palette,
                "color_name": color_name
            })
        except Exception as e:
            db.session.rollback()
            print(f"Error processing {filename}: {e}")
            continue

    return jsonify(processed_items)

@app.route('/wardrobe/item/<item_id>', methods=['DELETE'])
@limiter.limit("20 per minute")
def wardrobe_delete_item(item_id):
    owner_id = get_owner_id()
    item = WardrobeItem.query.filter_by(id=item_id, owner_id=owner_id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"error": "Item not found"}), 404

@app.route('/wardrobe/item/<item_id>', methods=['PATCH'])
@limiter.limit("20 per minute")
def wardrobe_update_item(item_id):
    owner_id = get_owner_id()
    data = request.get_json()
    new_category = data.get('category')

    item = WardrobeItem.query.filter_by(id=item_id, owner_id=owner_id).first()
    if item and new_category:
        item.category = new_category
        db.session.commit()
        return jsonify({"id": item.id, "category": item.category})

    return jsonify({"error": "Item not found or invalid data"}), 404

@app.route('/wardrobe/generate', methods=['POST'])
@limiter.limit("10 per minute")
def wardrobe_generate():
    owner_id = get_owner_id()
    items = WardrobeItem.query.filter_by(owner_id=owner_id).all()
    if not items:
        return jsonify({"error": "Wardrobe is empty"}), 400

    wardrobe_dict = {}
    for i in items:
        cat = i.category
        if cat not in wardrobe_dict: wardrobe_dict[cat] = []
        wardrobe_dict[cat].append({
            "id": i.id,
            "image_path": i.image_path,
            "category": i.category,
            "label": i.label,
            "accent_color": [int(x) for x in i.accent_rgb.split(',')],
            "color_name": i.color_name
        })

    recommended_hex_list = []
    avoid_hex_list = []
    if current_user.is_authenticated:
        palette_entry = UserPalette.query.get(current_user.id)
        if palette_entry and palette_entry.palette_json:
            try:
                palette_data = json.loads(palette_entry.palette_json)
                if "palette" in palette_data:
                    recommended_hex_list = [c["hex"] for c in palette_data["palette"].get("recommended", [])]
                    avoid_hex_list = [c["hex"] for c in palette_data["palette"].get("avoid", [])]
                else:
                    recommended_hex_list = [c["hex"] for c in palette_data.get("recommended", [])]
                    avoid_hex_list = [c["hex"] for c in palette_data.get("avoid", [])]
            except Exception as e:
                print(f"Error parsing UserPalette: {e}")
        elif current_user.skin_hex and current_user.undertone:
            try:
                l_user, _, _ = hex_to_lab(current_user.skin_hex)
                skin_tone = classify_skin_tone(l_user)
                undertone_base = current_user.undertone.split(" (")[0]
                fashion_palette = get_fashion_palette(skin_tone, undertone_base, "Medium")
                recommended_hex_list = [c["hex"] for c in fashion_palette.get("recommended", [])]
                avoid_hex_list = [c["hex"] for c in fashion_palette.get("avoid", [])]
            except Exception as e:
                print(f"Error constructing fallback fashion palette: {e}")

    combos = wardrobe_logic.generate_combinations(wardrobe_dict)
    ranked = wardrobe_logic.rank_combinations(combos, recommended_hex_list=recommended_hex_list, avoid_hex_list=avoid_hex_list)
    return jsonify(ranked)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No image selected'}), 400

    filename = secure_upload_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    if not validate_image_file(filepath):
        return jsonify({'error': 'Invalid image file'}), 400

    image = cv2.imread(filepath)
    if image is None: return jsonify({'error': 'Invalid image'}), 400

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks: return jsonify({'error': 'No face detected'}), 400

    face_landmarks_list = detection_result.face_landmarks[0]
    height, width, _ = image.shape

    face_shape_data = analyze_face_shape(face_landmarks_list, width, height)
    face_shape = face_shape_data['primary_shape']
    skin_hex, undertone, brightness, season_12, mst_label = analyze_color(image, face_landmarks_list)

    current_user.face_shape = face_shape
    current_user.skin_hex = skin_hex
    current_user.undertone = undertone
    db.session.commit()

    face_shape_base = face_shape.split(" (")[0]
    grooming_rec = GROOMING_RECOMMENDATIONS.get(face_shape_base, {})
    skin_tone_cat = classify_skin_tone(brightness[0])
    undertone_base = undertone.split(" (")[0]
    contrast_val = analyze_contrast(image, face_landmarks_list)
    metrics_data = compute_face_metrics(face_landmarks_list, width, height)
    eyewear_rec, _ = select_eyewear_frames(face_shape_base, metrics_data["face_size"], metrics_data["nose_width"], metrics_data["eye_spacing"], undertone_base, contrast_val)
    fashion_palette = get_fashion_palette(skin_tone_cat, undertone_base, contrast_val)

    f_query = Foundation.query.filter_by(category='Foundation').all()
    nearest_f = "None"
    if f_query:
        f_labs = np.array([[f.l, f.a, f.b] for f in f_query])
        dists = delta_e_cie2000_vec(brightness, f_labs)
        best_idx = np.argmin(dists)
        nearest_f = f"{f_query[best_idx].brand} {f_query[best_idx].product} ({f_query[best_idx].name})"

    recs = {
        "description": RECOMMENDATIONS.get(face_shape_base, ""),
        "foundation": nearest_f,
        "hair": grooming_rec.get("hair_male", []) if current_user.gender == 'male' else grooming_rec.get("hair_female", []),
        "beard": grooming_rec.get("beard", []) if current_user.gender == 'male' else [],
        "glasses": [g["name"] for g in eyewear_rec],
        "palette": [c["hex"] for c in fashion_palette.get("recommended", [])[:4]]
    }

    return jsonify({
        "face_shape": face_shape,
        "skin_hex": skin_hex,
        "undertone": undertone,
        "season": season_12,
        "mst_level": mst_label,
        "landmarks": [[lm.x, lm.y] for lm in face_landmarks_list],
        "recommendations": recs
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)