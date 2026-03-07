import os
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from reference_data import MST_LAB, SEASON_LAB, get_mst_label

app = Flask(__name__)
app.config['SECRET_KEY'] = 'slayr-secret-key-change-this-in-prod'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///slayr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = 'face_landmarker.task'


# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    full_name = db.Column(db.String(150), nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    password = db.Column(db.String(150), nullable=False)
    # Stored Profile Data
    face_shape = db.Column(db.String(50))
    skin_hex = db.Column(db.String(20))
    undertone = db.Column(db.String(20))

class Foundation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(100))
    product = db.Column(db.String(200)) # e.g., "Luminous Foundation"
    name = db.Column(db.String(100))    # e.g., "355N"
    hex_code = db.Column(db.String(20))
    # CIELAB Standard Coordinates
    l = db.Column(db.Float)
    a = db.Column(db.Float)
    b = db.Column(db.Float)
    category = db.Column(db.String(50)) # Foundation or Concealer
    brand_tier = db.Column(db.String(50)) # High-End or Drugstore
    hue = db.Column(db.String(50))      # For penalty logic
    image_url = db.Column(db.String(500))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Stylist AI Logic (Mini ChatGPT for Style) ---
SILA_KNOWLEDGE = {
    "face": "Your facial architecture is the blueprint of your style. I recommend balancing your proportions—for example, angular frames soften a round face, while rounded styles complement a square jaw. Since you're in Slayr, our Morpho-Face analysis accurately maps your geometry to find exactly what harmonizes with your bone structure.",
    "color": "Colors resonate with your natural undertone. If you're cool-toned, embrace silvers and jewel tones. Warm-toned individuals shine in gold, mocha, and olive greens. Our Vibe-Check feature can reveal your specific seasonal spectrum!",
    "match": "Cosmetic matching is about finding your skin's perfect frequency. Aim for foundations that disappear into your neck line—never settle for 'almost'. Smart-Match uses AI to scan your tone and find the exact hex twins for your skin.",
    "wardrobe": "A capsule wardrobe is about high-frequency pieces that mix seamlessly. Start with architectural basics: a crisp white shirt, perfectly tailored trousers, and a signature coat. Use our Capsule Wardrobe feature to curate a collection that echoes your aesthetic vibe.",
    "grooming": "Precision is key. Align your beard or hair style with your face shape to create structural harmony. A well-groomed signature is your best accessory. My Grooming Blueprint can map your follicle alignment for the perfect finish.",
    "glasses": "Finding the right frames is pure geometry. If you have an oval face, you're a canvas for almost any style. Square faces benefit from round or aviator shapes to soften the jaw. Check out Frame-Fit for a virtual synthesis of your ideal eyewear.",
    "makeup": "Makeup should amplify your resonance, not mask it. Match your foundation to your jawline and choose lip tones that harmonize with your seasonal palette—Vibe-Check can help you pick the perfect shades!",
    "hello": "Hi there! I'm Sila, your personal style synthesist. I've been analyzing the latest trends and architectural aesthetics. How can I help you refine your resonance today?",
    "who": "I am Sila, Slayr's resident AI Stylist. I'm trained on thousands of style archetypes, geometric ratios, and chromatic frequencies. My purpose is to help you decipher your unique aesthetic blueprint.",
    "thanks": "You're very welcome! It's my mission to see you looking your absolute best. Anything else on your mind?",
    "default": "That's an interesting aesthetic question! While I'm still evolving my style-engine, I'd suggest focusing on your natural architecture and choosing pieces that make you feel like the most refined version of yourself. Have you tried my specific analysis features in the sidebar?"
}

@app.route('/sila-chat', methods=['POST'])
def sila_chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()
    
    response = SILA_KNOWLEDGE["default"]
    
    # Priority keyword matching
    if any(k in user_message for k in ["thanks", "thank you", "thx"]):
        response = SILA_KNOWLEDGE["thanks"]
    elif any(k in user_message for k in ["hello", "hi", "hey", "hola"]):
        response = SILA_KNOWLEDGE["hello"]
    elif any(k in user_message for k in ["face", "shape", "morpho", "head"]):
        response = SILA_KNOWLEDGE["face"]
    elif any(k in user_message for k in ["color", "palette", "vibe", "undertone"]):
        response = SILA_KNOWLEDGE["color"]
    elif any(k in user_message for k in ["foundation", "match", "skin", "shade"]):
        response = SILA_KNOWLEDGE["match"]
    elif any(k in user_message for k in ["wardrobe", "clothes", "capsule", "outfit", "style"]):
        response = SILA_KNOWLEDGE["wardrobe"]
    elif any(k in user_message for k in ["beard", "hair", "grooming", "shave"]):
        response = SILA_KNOWLEDGE["grooming"]
    elif any(k in user_message for k in ["glasses", "frames", "eyewear", "specs"]):
        response = SILA_KNOWLEDGE["glasses"]
    elif any(k in user_message for k in ["makeup", "lipstick", "cosmetic", "beauty"]):
        response = SILA_KNOWLEDGE["makeup"]
    elif any(k in user_message for k in ["who", "what", "are you"]):
        response = SILA_KNOWLEDGE["who"]

    return jsonify({"response": response})

# --- Seeding ---
# Seed logic replaced by sync_shades.py


# --- MediaPipe Tasks Setup ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Lightweight MST / Season KNN (pure NumPy, no scikit-learn) ---
MST_X = np.array([item[:3] for item in MST_LAB], dtype=np.float32)
SEASON_X = np.array([item[:3] for item in SEASON_LAB], dtype=np.float32)
SEASON_Y = [item[3] for item in SEASON_LAB]

# --- Logic (Helpers: Face Shape) ---
RECOMMENDATIONS = {
        "Oval": "Balanced proportions. Lucky you! Most styles work.",
        "Square": "Strong jawline. Soften angles with waves or round frames.",
        "Round": "Soft angles. Add structure with angular frames or height in hair.",
        "Heart": "Wide forehead, narrow chin. Balance with chin-length bobs.",
        "Diamond": "Wide cheekbones. Highlight them, soften the chin.",
        "Oblong": "Long face. Add width with curls or oversized frames."
}

def analyze_face_shape(landmarks, image_width, image_height, matrix=None):
    coords = {}
    indices = [10, 152, 234, 454, 58, 288, 103, 332, 168, 6]
    
    if matrix is not None:
        # Use 3D Transformation Matrix for pose invariance (Anti-Gravity Refinement)
        inv_matrix = np.linalg.inv(matrix)
        for idx in indices:
            # MediaPipe landmarks are normalized [0,1], but matrix expects camera-space? 
            # Actually, standard MP Matrix works with normalized landmarks mapped to [-1, 1] or similar.
            # We'll use the matrix to project 2D landmarks into a canonical front-facing space.
            p = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z if hasattr(landmarks[idx], 'z') else 0, 1.0])
            p_canonical = inv_matrix @ p
            coords[idx] = p_canonical[:3]
    else:
        # Fallback to 2D ratios
        for idx in indices:
            coords[idx] = (landmarks[idx].x * image_width, landmarks[idx].y * image_height)

    def dist(p1_idx, p2_idx):
        return np.linalg.norm(np.array(coords[p1_idx]) - np.array(coords[p2_idx]))

    height = dist(10, 152)
    width_forehead = dist(103, 332)
    width_cheek = dist(234, 454)
    width_jaw = dist(58, 288)

    if height == 0: return "Oval"

    # Ratios
    h_w_ratio = height / width_cheek
    forehead_jaw_ratio = width_forehead / width_jaw
    cheek_jaw_ratio = width_cheek / width_jaw

    # Analysis logic
    if h_w_ratio > 1.3:
        if forehead_jaw_ratio > 1.1 and cheek_jaw_ratio > 1.1:
            return "Heart"
        return "Oblong"
    elif h_w_ratio < 0.95:
        if cheek_jaw_ratio < 1.1:
            return "Square"
        return "Round"
    else:
        # Balanced height/width
        if width_forehead > width_cheek * 0.9:
            if width_jaw > width_cheek * 0.9:
                return "Square"
            return "Heart"
        if width_cheek > width_forehead * 1.1:
            if width_cheek > width_jaw * 1.1:
                return "Diamond"
        return "Oval"

# --- Logic (Helpers: Color Analysis) ---

def delta_e_cie2000_vec(lab1, lab_array):
    """
    Vectorized Perceptual Distance.
    Balances Lightness (L) with Undertone (a, b) to prevent exact match errors.
    """
    dL = lab_array[:, 0] - lab1[0]
    da = lab_array[:, 1] - lab1[1]
    db = lab_array[:, 2] - lab1[2]
    
    # Balanced weights to ensure undertone accuracy alongside lightness
    return np.sqrt((dL * 1.5)**2 + (da * 1.8)**2 + (db * 1.8)**2)

def cv2_to_std_lab(l_cv2, a_cv2, b_cv2):
    """Converts OpenCV LAB (0-255) to Standard CIELAB (L:0-100, a/b:-128 to 127)"""
    l_std = (l_cv2 * 100.0) / 255.0
    a_std = a_cv2 - 128.0
    b_std = b_cv2 - 128.0
    return l_std, a_std, b_std

def analyze_color(image, landmarks, wrist_image=None):
    h, w, _ = image.shape
    
    # 1. Direct Skin Sampling - Focus on well-lit central areas
    rois = []
    # Sample from center forehead, under eyes, and upper cheeks (avoid jaw/edges which have shadows)
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
            
    if not rois: return "#D4A373", "Neutral", (60.0, 0.0, 0.0), "Soft Autumn", "MST 5"

    rois = np.array(rois)
    
    # Calculate luminance to filter out shadows
    luminance = 0.299 * rois[:, 2] + 0.587 * rois[:, 1] + 0.114 * rois[:, 0]
    
    # Keep the top 50% brightest patches to avoid shadows, then take the median of those
    threshold = np.percentile(luminance, 50)
    well_lit_rois = rois[luminance >= threshold]
    
    dominant_color = np.median(well_lit_rois, axis=0) 
    dominant_rgb = dominant_color[::-1].astype(int)
    hex_code = "#{:02x}{:02x}{:02x}".format(*dominant_rgb)
    
    # LAB Conversion: OpenCV (0-255)
    lab_color_cv2 = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2LAB)[0][0]
    
    l_raw = float(lab_color_cv2[0])
    a_raw = float(lab_color_cv2[1])
    b_raw = float(lab_color_cv2[2])
    
    # Normalize to Standard CIELAB (used by our AI models)
    l_std, a_std, b_std = cv2_to_std_lab(l_raw, a_raw, b_raw)
    
    # MST + Seasonal classification using lightweight KNN (no scikit-learn)
    # Nearest MST index
    diffs_mst = MST_X - np.array([l_std, a_std, b_std], dtype=np.float32)
    dists_mst = np.sqrt(np.sum(diffs_mst * diffs_mst, axis=1))
    mst_level = int(np.argmin(dists_mst))
    mst_label = get_mst_label(mst_level)

    # 3-NN majority vote for 12-season label
    diffs_season = SEASON_X - np.array([l_std, a_std, b_std], dtype=np.float32)
    dists_season = np.sqrt(np.sum(diffs_season * diffs_season, axis=1))
    k = min(3, len(SEASON_Y))
    nn_idx = np.argsort(dists_season)[:k]
    votes = {}
    for i in nn_idx:
        lab = SEASON_Y[int(i)]
        votes[lab] = votes.get(lab, 0) + 1
    season_12 = max(votes, key=votes.get) if votes else None
    
    # --- Undertone Extraction ---
    # Default to face's a/b values
    u_a, u_b = a_std, b_std
    
    # If wrist image provided, use it for pinpoint accuracy
    if wrist_image is not None:
        wh, ww, _ = wrist_image.shape
        # Sample the center 30% of the wrist image
        cy, cx = wh // 2, ww // 2
        ry, rx = int(wh * 0.15), int(ww * 0.15)
        wrist_patch = wrist_image[max(0, cy-ry):min(wh, cy+ry), max(0, cx-rx):min(ww, cx+rx)]
        
        if wrist_patch.size > 0:
            w_b = np.median(wrist_patch[:, :, 0])
            w_g = np.median(wrist_patch[:, :, 1])
            w_r = np.median(wrist_patch[:, :, 2])
            wrist_lab_cv2 = cv2.cvtColor(np.uint8([[[w_b, w_g, w_r]]]), cv2.COLOR_BGR2LAB)[0][0]
            _, u_a, u_b = cv2_to_std_lab(float(wrist_lab_cv2[0]), float(wrist_lab_cv2[1]), float(wrist_lab_cv2[2]))

    # Natural balance: Warm has high b_std compared to a_std
    # Human skin is inherently yellow-leaning. If a_std >= b_std, it's almost certainly Cool.
    undertone = "Neutral"
    if u_b > u_a + 4: undertone = "Warm"
    elif u_a >= u_b - 1: undertone = "Cool"
        
    return hex_code, undertone, (l_std, a_std, b_std), season_12, mst_label

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    # Draw key landmarks for visualization (Mesh style)
    for landmark in landmarks:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)
    return image

def swatch_face(image, landmarks, foundation_hex, concealer_hex):
    h, w, _ = image.shape
    
    # Define zones
    # Cheek for Foundation (Index 234)
    cheek_idx = 234
    cheek_pt = (int(landmarks[cheek_idx].x * w), int(landmarks[cheek_idx].y * h))
    
    # Under-eye for Concealer (Index 101 for left eye, 330 for right eye)
    # Using 101 as a representative point for under-eye area
    eye_idx = 101 
    eye_pt = (int(landmarks[eye_idx].x * w), int(landmarks[eye_idx].y * h))
    
    # Draw simple swatches (Circles with border)
    f_rgb = hex_to_rgb(foundation_hex)[::-1] # BGR
    c_rgb = hex_to_rgb(concealer_hex)[::-1] # BGR
    
    cv2.circle(image, cheek_pt, 25, f_rgb, -1)
    cv2.circle(image, cheek_pt, 27, (255, 255, 255), 2)
    cv2.putText(image, "Foundation", (cheek_pt[0]-40, cheek_pt[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.circle(image, eye_pt, 20, c_rgb, -1)
    cv2.circle(image, eye_pt, 22, (255, 255, 255), 2)
    cv2.putText(image, "Concealer", (eye_pt[0]-30, eye_pt[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image

PALETTES = {
    "Spring": {"colors": ["#FFD700", "#FF7F50", "#98FB98", "#E6E6FA"], "desc": "Fresh, bright, and warm colors."},
    "Summer": {"colors": ["#B0E0E6", "#D8BFD8", "#F08080", "#778899"], "desc": "Soft, cool, and muted colors."},
    "Autumn": {"colors": ["#D2691E", "#8B4513", "#556B2F", "#FF8C00"], "desc": "Rich, warm, and earthy tones."},
    "Winter": {"colors": ["#000080", "#DC143C", "#000000", "#FFFFFF"], "desc": "Sharp, cool, and vivid colors."},
    # 12-Season Specifics (subset)
    "Light Spring": {"colors": ["#FFFACD", "#E0FFFF", "#FFB6C1", "#98FB98"], "desc": "Light, warm, and clear colors."},
    "True Spring": {"colors": ["#FFD700", "#FF8C00", "#32CD32", "#FF4500"], "desc": "Warm, bright, and saturated colors."},
    "Bright Spring": {"colors": ["#FFFF00", "#00FF7F", "#FF1493", "#00BFFF"], "desc": "Highly saturated, warm, and clear."},
    "Light Summer": {"colors": ["#F0F8FF", "#E6E6FA", "#FFC0CB", "#AFEEEE"], "desc": "Light, cool, and delicate colors."},
    "True Summer": {"colors": ["#87CEEB", "#DDA0DD", "#B0C4DE", "#4682B4"], "desc": "Cool, muted, and soft colors."},
    "Soft Summer": {"colors": ["#778899", "#BC8F8F", "#BDB76B", "#6A5ACD"], "desc": "Muted, cool, and hazy colors."},
    "Soft Autumn": {"colors": ["#BC8F8F", "#BDB76B", "#CD853F", "#556B2F"], "desc": "Muted, warm, and gentle earthy tones."},
    "True Autumn": {"colors": ["#D2691E", "#8B4513", "#A52A2A", "#808000"], "desc": "Rich, warm, and deep earthy colors."},
    "Warm Autumn": {"colors": ["#FF8C00", "#D2691E", "#B8860B", "#A52A2A"], "desc": "Warm, deep, and glowing tones."},
    "Deep Winter": {"colors": ["#191970", "#800000", "#006400", "#000000"], "desc": "Deep, cool, and intense colors."},
    "True Winter": {"colors": ["#0000FF", "#FF0000", "#FFFFFF", "#000000"], "desc": "True cool, high contrast colors."},
    "Bright Winter": {"colors": ["#00FFFF", "#FF00FF", "#00FF00", "#FFFFFF"], "desc": "Bright, cool, and icy tones."}
}

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('username')
        password = request.form.get('password')
        # Check by username or email
        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('auth/login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        gender = request.form.get('gender')
        password = request.form.get('password')
        
        # Simple username handle from email
        username = email.split('@')[0] if email and '@' in email else email
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            # If username handle taken, append random or use full email
            username = email

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, email=email, full_name=full_name, gender=gender, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('auth/signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('features/dashboard.html', user=current_user)

@app.route('/chroma-skin', methods=['GET', 'POST'])
@login_required
def chroma_skin():
    skin_hex = None
    undertone = None
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            wrist_file = request.files.get('wrist_image')
            
            if file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                
                wrist_image = None
                if wrist_file and wrist_file.filename != '':
                    wrist_filepath = os.path.join(UPLOAD_FOLDER, 'wrist_' + wrist_file.filename)
                    wrist_file.save(wrist_filepath)
                    wrist_image = cv2.imread(wrist_filepath)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    skin_hex, undertone, lab_std, season_12, mst_label = analyze_color(image, detection_result.face_landmarks[0], wrist_image=wrist_image)
                    
                    # Update User Profile
                    current_user.skin_hex = skin_hex
                    current_user.undertone = undertone
                    db.session.commit()
                    
                    return render_template('features/chroma_skin.html', skin_hex=skin_hex, undertone=undertone, uploaded_image=file.filename, mst_label=mst_label)
                    
    return render_template('features/chroma_skin.html')

@app.route('/smart-match', methods=['GET', 'POST'])
@login_required
def smart_match():
    # Use distinct to get available brands for the filter
    brands = db.session.query(Foundation.brand).distinct().all()
    brands = sorted([b[0] for b in brands if b[0]])
    
    matches = None
    concealer_matches = None
    skin_hex = None
    swatched_image = None
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            selected_brand = request.form.get('brand')
            
            if file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    hex_code, undertone, lab_std, season_12, mst_label = analyze_color(image, detection_result.face_landmarks[0])
                    skin_hex = hex_code
                    l_user, a_user, b_user = lab_std
                    
                    # --- Foundation Match (Vectorized) ---
                    f_query = Foundation.query.filter_by(category='Foundation')
                    if selected_brand:
                        f_query = f_query.filter_by(brand=selected_brand)
                    
                    foundations = f_query.all()
                    if foundations:
                        f_labs = np.array([[f.l, f.a, f.b] for f in foundations])
                        
                        # Calculate Base Distances
                        distances = delta_e_cie2000_vec(lab_std, f_labs)
                        
                        # Sort and pick top 3
                        top_indices = np.argsort(distances)[:3]
                        matches = []
                        for rank, i in enumerate(top_indices):
                            f = foundations[i]
                            dist = distances[i]
                            # Confidence calculation: 100 - (dist * penalty)
                            confidence = max(1, min(99, int(100 - (dist * 3.5))))
                            f.match_score = confidence
                            matches.append(f)

                    # --- Concealer Match (Lightness +5.0) ---
                    # Logic: Target L +5.0 (brightening), identical undertone
                    target_l = min(100.0, l_user + 5.0)
                    
                    c_query = Foundation.query.filter_by(category='Concealer')
                    if selected_brand:
                        c_query = c_query.filter_by(brand=selected_brand)
                    concealers = c_query.all()
                    
                    if concealers:
                        c_labs = np.array([[c.l, c.a, c.b] for c in concealers])
                        
                        # Distance to the ideal brightening concealer
                        c_dists = delta_e_cie2000_vec([target_l, a_user, b_user], c_labs)
                        
                        top_c_indices = np.argsort(c_dists)[:3]
                        concealer_matches = []
                        for rank, i in enumerate(top_c_indices):
                            c = concealers[i]
                            dist = c_dists[i]
                            confidence = max(1, min(99, int(100 - (dist * 3.5))))
                            c.match_score = confidence
                            concealer_matches.append(c)
                    
                    # --- Visual Swatching (Reuse updated points) ---
                    swatch_coords = {}
                    if detection_result.face_landmarks:
                        landmarks = detection_result.face_landmarks[0]
                        # Jawline for foundation
                        f_swatch_idx = 58 
                        swatch_coords['foundation'] = {'x': landmarks[f_swatch_idx].x, 'y': landmarks[f_swatch_idx].y}
                        # Under-eye for concealer
                        c_swatch_idx = 101
                        swatch_coords['concealer'] = {'x': landmarks[c_swatch_idx].x, 'y': landmarks[c_swatch_idx].y}
                        
                        output_path = os.path.join(UPLOAD_FOLDER, 'clean_' + file.filename)
                        cv2.imwrite(output_path, image)
                        swatched_image = 'clean_' + file.filename
                    
                    return render_template('features/smart_match.html', brands=brands, matches=matches, concealer_matches=concealer_matches, skin_hex=skin_hex, swatched_image=swatched_image, swatch_coords=swatch_coords, season=season_12, mst_label=mst_label)

    return render_template('features/smart_match.html', brands=brands, matches=matches)

@app.route('/morpho-face')
@login_required
def morpho_face():
    # Show "Unknown" initially to hide results until analysis
    return render_template('features/morpho_face.html', shape="Unknown", description="")

@app.route('/vibe-check', methods=['GET', 'POST'])
@login_required
def vibe_check():
    season = None
    palette = None
    uploaded_image = None
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                uploaded_image = file.filename
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    # Reuse analyze_color logic
                    skin_hex, undertone, lab_std, season, mst_label = analyze_color(image, detection_result.face_landmarks[0])
                    
                    palette = PALETTES.get(season, PALETTES["Winter"])
                    
                    # Update User Profile
                    current_user.undertone = undertone
                    current_user.skin_hex = skin_hex
                    db.session.commit()
    
    return render_template('features/vibe_check.html', season=season, palette=palette, uploaded_image=uploaded_image)
@app.route('/morpho-analyze', methods=['POST'])
@login_required
def morpho_analyze():
    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No image selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    image = cv2.imread(filepath)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks[0]
        # Extract Transformation Matrix for Anti-Gravity De-rotation
        matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None
        height, width, _ = image.shape
        face_shape = analyze_face_shape(face_landmarks_list, width, height, matrix)
        
        # Update User
        current_user.face_shape = face_shape
        db.session.commit()
        
        # Draw Mesh
        image = draw_landmarks(image, face_landmarks_list)
        output_path = os.path.join(UPLOAD_FOLDER, 'morpho_' + file.filename)
        cv2.imwrite(output_path, image)
        
        return jsonify({
            'shape': face_shape,
            'description': RECOMMENDATIONS.get(face_shape),
            'image_url': url_for('static', filename='uploads/morpho_' + file.filename)
        })
        
    return jsonify({'error': 'No face detected'}), 400

# --- Overlay Helper ---
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)
        h, w, _ = overlay.shape

    if x + w > bg_w: w = bg_w - x
    if y + h > bg_h: h = bg_h - y
    # --- Robust Transparency Handling ---
    # If the image is 3-channel, or 4-channel but fully opaque, we need to generate a mask.
    has_alpha = False
    if overlay.shape[2] == 4:
        # Check if alpha channel has any variation (not just all 255)
        if np.min(overlay[:, :, 3]) < 255:
            has_alpha = True
            
    if not has_alpha:
        # Convert to BGRA if needed
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
            
        # Create a mask based on luminance (assuming light background)
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGRA2GRAY)
        # Threshold: Any pixel brighter than 230 is considered background (transparent)
        # We also treat the specific "checkerboard" gray (around 204-240) as background if needed, 
        # but a simple high-pass threshold usually catches white/light boxes.
        _, mask_bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # refinement: erode/dilate to clean up edges of the mask
        kernel = np.ones((3,3), np.uint8)
        mask_bw = cv2.erode(mask_bw, kernel, iterations=1)
        # mask_bw = cv2.dilate(mask_bw, kernel, iterations=1) # Optional if too thin
        
        # Assign this generated mask to the alpha channel
        overlay[:, :, 3] = mask_bw
        
    # --- Standard Blending ---
    overlay_img = overlay[:h, :w, :3]
    mask = overlay[:h, :w, 3] / 255.0

    # Ensure mask is 3-channel for multiplication
    mask = cv2.merge([mask, mask, mask])

    # Convert to float for blending
    background_roi = background[y:y+h, x:x+w].astype(float)
    overlay_img = overlay_img.astype(float)
    
    # Correct Blending Formula: BG * (1 - Alpha) + FG * Alpha
    blended = (background_roi * (1.0 - mask)) + (overlay_img * mask)
    
    background[y:y+h, x:x+w] = blended.astype('uint8')
    
    return background

@app.route('/frame-fit', methods=['GET', 'POST'])
@login_required
def frame_fit():
    processed_image = None
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    h, w, _ = image.shape
                    
                    # Left Eye (33), Right Eye (263)
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    
                    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
                    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
                    
                    eye_width = np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry]))
                    glasses_width = int(eye_width * 2.5) # Scale factor
                    
                    glasses_path = 'static/assets/glasses.png' # Placeholder
                    if os.path.exists(glasses_path):
                        glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
                        aspect_ratio = glasses.shape[0] / glasses.shape[1]
                        glasses_height = int(glasses_width * aspect_ratio)
                        
                        center_x = (lx + rx) // 2
                        center_y = (ly + ry) // 2
                        
                        # Adjust for slight head tilt
                        angle = np.degrees(np.arctan2(ry - ly, rx - lx))
                        
                        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
                        glasses = cv2.warpAffine(glasses, M, (glasses.shape[1], glasses.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                        
                        top_left_x = center_x - (glasses_width // 2)
                        top_left_y = center_y - (glasses_height // 2)
                        
                        image = overlay_transparent(image, glasses, top_left_x, top_left_y, (glasses_width, glasses_height))
                
                output_path = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
                cv2.imwrite(output_path, image)
                processed_image = 'processed_' + file.filename

    return render_template('features/frame_fit.html', processed_image=processed_image)

@app.route('/grooming-blueprint', methods=['GET', 'POST'])
@login_required
def grooming_blueprint():
    processed_image = None
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    h, w, _ = image.shape
                    
                    # Mouth Center (13) & Chin (152) & Jawline
                    # Rough sizing based on jaw width
                    jaw_left = landmarks[234]
                    jaw_right = landmarks[454]
                    chin = landmarks[152]
                    
                    jl_x, jl_y = int(jaw_left.x * w), int(jaw_left.y * h)
                    jr_x, jr_y = int(jaw_right.x * w), int(jaw_right.y * h)
                    c_x, c_y = int(chin.x * w), int(chin.y * h)
                    
                    jaw_width = np.linalg.norm(np.array([jl_x, jl_y]) - np.array([jr_x, jr_y]))
                    beard_width = int(jaw_width * 1.2)
                    
                    beard_path = 'static/assets/beard.png' # Need to ensure this exists
                    if os.path.exists(beard_path):
                        beard = cv2.imread(beard_path, cv2.IMREAD_UNCHANGED)
                        aspect_ratio = beard.shape[0] / beard.shape[1]
                        beard_height = int(beard_width * aspect_ratio)
                        
                        # Use mouth center as anchor (Index 13)
                        mouth_center = landmarks[13]
                        mx, my = int(mouth_center.x * w), int(mouth_center.y * h)
                        
                        # Adjust center based on face pitch (rough estimate from chin position relative to mouth)
                        pitch_offset = (c_y - my) * 0.1
                        center_x = mx
                        center_y = int(my + (beard_height * 0.4) + pitch_offset)
                        
                        top_left_x = center_x - (beard_width // 2)
                        top_left_y = center_y - (beard_height // 2)
                        
                        image = overlay_transparent(image, beard, top_left_x, top_left_y, (beard_width, beard_height))
                
                output_path = os.path.join(UPLOAD_FOLDER, 'groomed_' + file.filename)
                cv2.imwrite(output_path, image)
                processed_image = 'groomed_' + file.filename

    return render_template('features/grooming.html', processed_image=processed_image)

# --- Style DNA Data ---
STYLE_DATA = {
    "Oval": {"lines": "Balanced", "styles": ["Versatile", "Timeless"]},
    "Square": {"lines": "Softened", "styles": ["Large curves", "Unstructured tops"]},
    "Round": {"lines": "Angular", "styles": ["Vertical lines", "Structured jackets"]},
    "Heart": {"lines": "Bottom-weighted", "styles": ["A-line skirts", "Wide-leg pants"]},
    "Diamond": {"lines": "Cheek-focused", "styles": ["Boat necks", "V-necks"]},
    "Oblong": {"lines": "Horizontal", "styles": ["Wide belts", "Oversized frames"]}
}

@app.route('/capsule-wardrobe', methods=['GET', 'POST'])
@login_required
def capsule_wardrobe():
    vibe = None
    items = []
    
    # Personalization modifiers
    base_season = current_user.undertone if current_user.undertone else "Neutral"
    if base_season == "Warm": user_palette = PALETTES["Autumn"]
    elif base_season == "Cool": user_palette = PALETTES["Winter"]
    else: user_palette = PALETTES["Spring"]
    
    accent_color = user_palette["colors"][0]
    face_shape = current_user.face_shape if current_user.face_shape else "Oval"
    style_info = STYLE_DATA.get(face_shape, STYLE_DATA["Oval"])

    if request.method == 'POST':
        vibe = request.form.get('vibe')
        
        # Dynamic Item Generation based on Face Shape "Lines" and Season "Colors"
        if vibe == 'Casual':
            items = [
                {'name': f'{style_info["lines"]} Fit Blazer', 'type': 'Outerwear', 'icon': '🧥', 'color': user_palette["colors"][1]},
                {'name': 'Straight Leg Jeans', 'type': 'Bottoms', 'icon': '👖', 'color': '#4A5568'},
                {'name': f'{style_info["styles"][0]} Top', 'type': 'Tops', 'icon': '👕', 'color': accent_color},
                {'name': 'Chunky Loafers', 'type': 'Shoes', 'icon': '👞', 'color': '#2D3748'}
            ]
        elif vibe == 'Business':
            items = [
                {'name': f'{style_info["styles"][1]} Blouse', 'type': 'Tops', 'icon': '👚', 'color': accent_color},
                {'name': 'Tailored Trousers', 'type': 'Bottoms', 'icon': '👖', 'color': user_palette["colors"][2]},
                {'name': 'Structured Tote', 'type': 'Accessories', 'icon': '👜', 'color': '#1A202C'},
                {'name': 'Pointed Heels', 'type': 'Shoes', 'icon': '👠', 'color': '#000000'}
            ]
        elif vibe == 'Party':
            items = [
                {'name': 'Statement Dress', 'type': 'Dresses', 'icon': '👗', 'color': accent_color},
                {'name': f'{style_info["lines"]} Cut Coat', 'type': 'Outerwear', 'icon': '🧥', 'color': user_palette["colors"][3]},
                {'name': 'Clutch Bag', 'type': 'Accessories', 'icon': '👛', 'color': '#FFD700'},
                {'name': 'Stiletto Boots', 'type': 'Shoes', 'icon': '👢', 'color': '#000000'}
            ]
            
    return render_template('features/wardrobe.html', vibe=vibe, items=items, accent=accent_color, face_shape=face_shape)

@app.route('/consultation')
@login_required
def consultation():
    return render_template('features/consultation.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No image selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None: return jsonify({'error': 'Invalid image'}), 400
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks: return jsonify({'error': 'No face detected'}), 400
    
    face_landmarks_list = detection_result.face_landmarks[0]
    height, width, _ = image.shape
    
    face_shape = analyze_face_shape(face_landmarks_list, width, height)
    skin_hex, undertone, brightness, season_12, mst_label = analyze_color(image, face_landmarks_list)
    
    current_user.face_shape = face_shape
    current_user.skin_hex = skin_hex
    current_user.undertone = undertone
    db.session.commit()
    
    return jsonify({
        "face_shape": face_shape,
        "skin_hex": skin_hex,
        "undertone": undertone,
        "season": season_12,
        "mst_level": mst_label,
        "landmarks": [[lm.x, lm.y] for lm in face_landmarks_list]
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
