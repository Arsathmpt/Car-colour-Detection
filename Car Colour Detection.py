import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import colorsys
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarLens · AI Vehicle Inspector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Injected CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --bg:        #0A0A0F;
  --surface:   #111118;
  --surface2:  #1A1A26;
  --border:    #2A2A3E;
  --accent:    #FF4D6D;
  --accent2:   #00F5C4;
  --accent3:   #FFB347;
  --text:      #F0F0FF;
  --muted:     #6B6B8A;
  --card-glow: 0 0 40px rgba(255,77,109,0.08);
}

html, body, [class*="css"] {
  font-family: 'DM Mono', monospace;
  background-color: var(--bg);
  color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero header ── */
.hero {
  background: linear-gradient(135deg, #0A0A0F 0%, #12121F 60%, #1A0A14 100%);
  border-bottom: 1px solid var(--border);
  padding: 3rem 4rem 2.5rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -80px; right: -80px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(255,77,109,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -60px; left: 30%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(0,245,196,0.07) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.25em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 5vw, 4rem);
  font-weight: 800;
  line-height: 1.05;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  color: var(--muted);
  font-size: 0.88rem;
  max-width: 520px;
  line-height: 1.7;
  margin-top: 0.5rem;
}
.badge-row {
  display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.5rem;
}
.badge {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.25rem 0.85rem;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: var(--muted);
  background: var(--surface);
}
.badge.hot { border-color: var(--accent); color: var(--accent); }
.badge.teal { border-color: var(--accent2); color: var(--accent2); }

/* ── Main canvas ── */
.main-area {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  min-height: calc(100vh - 220px);
}

/* ── Upload zone ── */
.upload-zone {
  border-right: 1px solid var(--border);
  padding: 2.5rem 3rem;
  background: var(--surface);
}
.section-label {
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 1.2rem;
}

/* ── Results panel ── */
.results-zone {
  padding: 2.5rem 3rem;
  background: var(--bg);
}

/* ── Stat cards ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1.5rem;
}
.stat-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent); }
.stat-card::before {
  content: '';
  position: absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1;
}
.stat-label {
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* ── Colour chip ── */
.color-row {
  display: flex; align-items: center; gap: 1rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1.2rem;
  margin-bottom: 0.6rem;
  transition: border-color 0.2s;
}
.color-row:hover { border-color: var(--border); filter: brightness(1.05); }
.color-swatch {
  width: 36px; height: 36px;
  border-radius: 8px;
  flex-shrink: 0;
  border: 2px solid rgba(255,255,255,0.08);
}
.color-name {
  font-family: 'Syne', sans-serif;
  font-size: 0.95rem;
  font-weight: 700;
}
.color-count {
  margin-left: auto;
  font-size: 0.7rem;
  color: var(--muted);
  letter-spacing: 0.1em;
}
.conf-pill {
  font-size: 0.65rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: rgba(0,245,196,0.1);
  color: var(--accent2);
  border: 1px solid rgba(0,245,196,0.2);
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
  background: var(--surface2) !important;
  padding: 2rem !important;
  transition: border-color 0.3s;
}
.stFileUploader > div:hover {
  border-color: var(--accent) !important;
}
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  padding: 0.75rem 2rem !important;
  letter-spacing: 0.05em !important;
  width: 100% !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
div[data-testid="stImage"] img { border-radius: 14px; width: 100%; }
.stSuccess {
  background: rgba(0,245,196,0.08) !important;
  border: 1px solid rgba(0,245,196,0.25) !important;
  border-radius: 10px !important;
  color: var(--accent2) !important;
}
.stWarning {
  background: rgba(255,179,71,0.08) !important;
  border: 1px solid rgba(255,179,71,0.25) !important;
  border-radius: 10px !important;
  color: var(--accent3) !important;
}
.stInfo {
  background: rgba(255,77,109,0.07) !important;
  border: 1px solid rgba(255,77,109,0.2) !important;
  border-radius: 10px !important;
  color: #FF8FA3 !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Colour classification ─────────────────────────────────────────────────────
COLOR_PALETTE = {
    "Red":    ([0, 70, 50], [10, 255, 255]),
    "Red2":   ([170, 70, 50], [180, 255, 255]),
    "Orange": ([11, 100, 100], [25, 255, 255]),
    "Yellow": ([26, 80, 100], [34, 255, 255]),
    "Green":  ([35, 60, 40], [85, 255, 255]),
    "Cyan":   ([86, 60, 60], [100, 255, 255]),
    "Blue":   ([101, 60, 40], [130, 255, 255]),
    "Purple": ([131, 40, 40], [160, 255, 255]),
    "Pink":   ([161, 30, 100], [169, 255, 255]),
    "White":  ([0, 0, 180], [180, 30, 255]),
    "Black":  ([0, 0, 0], [180, 255, 50]),
    "Silver": ([0, 0, 110], [180, 25, 200]),
    "Gray":   ([0, 0, 51], [180, 25, 180]),
}

COLOR_HEX = {
    "Red":    "#E53935", "Red2":   "#E53935",
    "Orange": "#FB8C00", "Yellow": "#FDD835",
    "Green":  "#43A047", "Cyan":   "#00ACC1",
    "Blue":   "#1E88E5", "Purple": "#8E24AA",
    "Pink":   "#F06292", "White":  "#F5F5F5",
    "Black":  "#212121", "Silver": "#B0BEC5",
    "Gray":   "#78909C",
}

def classify_color(roi_bgr: np.ndarray) -> tuple[str, float]:
    """Return (color_name, confidence) using HSV histogram analysis."""
    if roi_bgr.size == 0:
        return "Unknown", 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    best_name, best_pct = "Unknown", 0.0
    total_px = hsv.shape[0] * hsv.shape[1]

    for name, (lo, hi) in COLOR_PALETTE.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        pct = mask.sum() / 255 / total_px
        if pct > best_pct:
            best_pct, best_name = pct, name

    # Merge Red / Red2
    if best_name == "Red2":
        best_name = "Red"

    confidence = min(best_pct * 2.5, 1.0)  # scale to ~0-1
    return best_name, round(confidence * 100, 1)


@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov8n.pt")   # tiny – fast first load, accurate enough


def draw_boxes(img_bgr, detections):
    """Draw sleek bounding boxes on a copy of the image."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2), label, color_name, conf in detections:
        hex_col  = COLOR_HEX.get(color_name, "#FFFFFF")
        # Convert hex → BGR
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        box_color = (b, g, r)

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        tag = f"{label} · {color_name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(out, tag, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)
    return out


def run_detection(image_rgb: np.ndarray, model):
    """Run YOLOv8 detection, return annotated image + stats."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results  = model(img_bgr, conf=0.35, verbose=False)[0]

    detections = []
    color_tally: dict[str, int] = {}
    car_count = person_count = other_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "car":
            car_count += 1
            roi = img_bgr[y1:y2, x1:x2]
            color_name, col_conf = classify_color(roi)
            color_tally[color_name] = color_tally.get(color_name, 0) + 1
            detections.append(((x1,y1,x2,y2), "Car", color_name, col_conf))
        elif label == "person":
            person_count += 1
            detections.append(((x1,y1,x2,y2), "Person", "—", 0))
        else:
            other_count += 1
            detections.append(((x1,y1,x2,y2), label.title(), "—", 0))

    annotated = draw_boxes(img_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, {
        "cars":   car_count,
        "people": person_count,
        "others": other_count,
        "colors": color_tally,
        "total":  len(detections),
    }


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">// AI · COMPUTER VISION · v2.0</div>
  <div class="hero-title">Car<span>Lens</span></div>
  <div class="hero-sub">
    YOLOv8-powered vehicle detection with real-time HSV colour classification.
    Drop any traffic image and get instant insights.
  </div>
  <div class="badge-row">
    <span class="badge hot">YOLOv8n</span>
    <span class="badge teal">HSV Colour Analysis</span>
    <span class="badge">13 Colour Classes</span>
    <span class="badge">Real-Time</span>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-label">01 · Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Source Image", use_column_width=True)
        run_btn = st.button("⚡  Run Detection")
    else:
        st.info("Upload a traffic or street-scene photo to begin.")
        run_btn = False

with col_right:
    st.markdown('<div class="section-label">02 · Detection Results</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        model = load_model()
        img_arr = np.array(image)

        with st.spinner("Analysing with YOLOv8 …"):
            t0 = time.time()
            annotated, stats = run_detection(img_arr, model)
            elapsed = round((time.time() - t0) * 1000)

        st.image(annotated, caption="Annotated Output", use_column_width=True)
        st.success(f"✓ Detection complete in {elapsed} ms")

        # ── Stat cards ──
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-num">{stats['cars']}</div>
            <div class="stat-label">Cars Found</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{stats['people']}</div>
            <div class="stat-label">People Found</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{stats['total']}</div>
            <div class="stat-label">Total Objects</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if stats["colors"]:
            st.markdown('<div class="section-label">03 · Colour Breakdown</div>', unsafe_allow_html=True)
            sorted_colors = sorted(stats["colors"].items(), key=lambda x: -x[1])
            for color_name, count in sorted_colors:
                hex_col = COLOR_HEX.get(color_name, "#888888")
                st.markdown(f"""
                <div class="color-row">
                  <div class="color-swatch" style="background:{hex_col};"></div>
                  <div>
                    <div class="color-name">{color_name}</div>
                  </div>
                  <div class="color-count">{count} car{"s" if count>1 else ""}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No cars detected. Try an image with visible vehicles.")

    elif not uploaded:
        st.markdown("""
        <div style="
          height:320px; display:flex; flex-direction:column;
          align-items:center; justify-content:center;
          border: 1px dashed #2A2A3E; border-radius:16px;
          color:#6B6B8A; gap:1rem;
        ">
          <div style="font-size:3rem;">🚗</div>
          <div style="font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase;">
            Awaiting image…
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="
  border-top:1px solid #2A2A3E; padding:1.5rem 4rem;
  display:flex; justify-content:space-between; align-items:center;
  color:#6B6B8A; font-size:0.7rem; letter-spacing:0.1em;
">
  <span>CARLENS · Built by Mohamed Arsath</span>
  <span>YOLOv8 · OpenCV · Streamlit</span>
</div>
""", unsafe_allow_html=True)
