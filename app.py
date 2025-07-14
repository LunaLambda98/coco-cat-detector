import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from pathlib import Path
import time

# ============================== Page Configuration ==============================
st.set_page_config(
    page_title="🐱 Coco Cat Detector",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================== Custom CSS for Cute Styling ==============================
st.markdown("""
<style>
    /* Main theme colors - warm orange palette */
    .stApp {
        background: linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #ff9800, #ff5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cute card styling */
    .cute-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(255, 152, 0, 0.2);
        border: 3px solid #ffcc02;
        margin: 1rem 0;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 3px dashed #ff9800;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 193, 7, 0.1);
        transition: all 0.3s ease;
    }
    
    /* Success/Error message styling */
    .success-msg {
        background: linear-gradient(45deg, #4caf50, #8bc34a);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
    }
    
    .error-msg {
        background: linear-gradient(45deg, #ff5722, #f44336);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff9800, #ff5722);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 152, 0, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #ff9800, #ff5722);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #fff3e0 0%, #ffe0b2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================== Load Models (with caching) ==============================
@st.cache_resource
def load_models():
    """Load YOLO models with caching for better performance."""
    try:
        cat_detector = YOLO("yolov8n.pt")
         
        coco_classifier = YOLO("best.pt") 
        return cat_detector, coco_classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# ============================== Detection Functions ==============================
def detect_cats(image_array, cat_detector, conf_thres=0.5):
    """Enhanced cat detection with multi-scale optimization."""
    h, w = image_array.shape[:2]
    
    # Save image temporarily for YOLO processing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, image_array)
        temp_path = tmp_file.name
    
    cats = []
    detection_configs = [
        (640, conf_thres, "standard"),
        (640, 0.25, "low_conf"),
        (480, 0.25, "small_scale"),
        (800, 0.25, "large_scale"),
        (320, 0.2, "very_small_scale"),
    ]
    
    for imgsz, conf_thresh, strategy_name in detection_configs:
        if cats:
            break
            
        results = cat_detector(temp_path, imgsz=imgsz, conf=conf_thresh)
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                if cls_id == 15:  # cat class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cats.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "detection_config": strategy_name
                    })
        
        if cats:
            break
    
    # Clean up temporary file
    try:
        os.unlink(temp_path)
    except:
        pass
    
    return cats

def is_orange_color_in_region(image, bbox, orange_ratio_thres=0.35):
    """Check if region contains orange colors."""
    x1, y1, x2, y2 = bbox
    
    # Crop to center 70% to reduce background influence
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * 0.15), int(h * 0.15)
    cx1, cy1 = max(0, x1 + dx), max(0, y1 + dy)
    cx2, cy2 = min(image.shape[1], x2 - dx), min(image.shape[0], y2 - dy)
    roi = image[cy1:cy2, cx1:cx2]
    
    if roi.size == 0:
        return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    orange_mask = (
        (h >= 5) & (h <= 25) &
        (s > 50) &
        (v > 50)
    )
    
    total_px = roi.shape[0] * roi.shape[1]
    orange_px = int(np.sum(orange_mask))
    ratio = orange_px / total_px
    return ratio > orange_ratio_thres

def detect_coco_cat_streamlit(image_array, cat_detector, coco_classifier):
    """Main detection pipeline for Streamlit application."""
    h, w = image_array.shape[:2]
    
    # Stage 1: Cat detection
    cats = detect_cats(image_array, cat_detector)
    
    # Return immediately if no cats are detected
    if len(cats) == 0:
        return {
            "result": "no_cat",
            "confidence": 0.0,
            "message": "No cats detected in the image! 😿"
        }
    
    # Stage 2: Orange color filter
    orange_cats = [c for c in cats if is_orange_color_in_region(image_array, c["bbox"])]
    
    if len(orange_cats) == 0:
        return {
            "result": "no_orange_cat",
            "confidence": 0.0,
            "total_cats": len(cats),
            "message": "Found cats, but none are orange! 🐱"
        }
    
    # Stage 3: Coco classification - crop each orange cat and test
    is_coco = False
    max_conf = 0.0
    best_cat_info = None
    
    for idx, cat in enumerate(orange_cats):
        x1, y1, x2, y2 = cat["bbox"]
        
        # Crop the cat region
        cat_crop = image_array[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if cat_crop.shape[0] < 20 or cat_crop.shape[1] < 20:
            continue
            
        # Save crop temporarily and run classification
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, cat_crop)
            crop_results = coco_classifier(tmp_file.name)
            
            # Process results from the model
            for res in crop_results:
                # Handle classification model output (if res.probs exists)
                if res.probs is not None:
                    cls_id = int(res.probs.top1)
                    conf = float(res.probs.top1conf)
                    if conf > max_conf:
                        max_conf = conf
                        is_coco = (cls_id == 0)
                        best_cat_info = f"Classification: Cat {idx+1}, confidence {conf:.3f}"
                
                # Handle detection model output (if res.boxes exists)
                elif res.boxes is not None:
                    for box in res.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        if conf > max_conf:
                            max_conf = conf
                            is_coco = (cls_id == 0)  # 0 = Coco, 1 = not Coco
                            best_cat_info = f"Detection: Cat {idx+1}, confidence {conf:.3f}"
            
            # Clean up temp file
            os.unlink(tmp_file.name)
    
    return {
        "result": "is_coco" if is_coco else "not_coco",
        "confidence": max_conf,
        "total_cats": len(cats),
        "orange_cats": len(orange_cats),
        "detection_method": "standard",  # No fallback mode, standard detection only
        "best_detection": best_cat_info,
        "message": f"{'🎉 This is Coco!' if is_coco else '😸 This is an orange cat, but not Coco!'}"
    }

def draw_cute_results(image_array, cats, result):
    """Draw cute bounding boxes and annotations on the image."""
    img_copy = image_array.copy()
    
    for cat in cats:
        x1, y1, x2, y2 = cat["bbox"]
        
        # Make sure coordinates are valid
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_copy.shape[1], x2), min(img_copy.shape[0], y2)
        
        # Draw cute rectangle
        color = (0, 165, 255) if result["result"] in ["is_coco", "not_coco"] else (255, 100, 100)  # BGR format
        thickness = 3
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Add cute label
        label = "Orange Cat" if result["result"] in ["is_coco", "not_coco"] else "Cat"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background rectangle
        label_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(img_copy, (x1, label_y - text_height - 10), (x1 + text_width + 10, label_y + baseline), color, -1)
        
        # Draw text
        cv2.putText(img_copy, label, (x1 + 5, label_y - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        # Add confidence score
        conf_text = f"Conf: {cat.get('confidence', 0):.2f}"
        cv2.putText(img_copy, conf_text, (x1 + 5, y2 - 10), font, 0.5, color, 1)
    
    return img_copy

# ============================== Streamlit App ==============================
def main():
    # Header
    st.markdown('<h1 class="main-header">🐱 Coco Cat Detection System 🍊</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="cute-card">
        <h3 style="text-align: center; color: #ff9800;">🌟 Welcome to Coco's Smart Detection System! 🌟</h3>
        <p style="text-align: center; font-size: 1.1rem; color: #666;">
            Upload a photo and I'll help you find out if it's our lovely orange cat Coco! 
            <br>🔍 I can detect cats, check if they're orange, and identify our special Coco! 🧡
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🤖 Loading AI models... Please wait!"):
        cat_detector, coco_classifier = load_models()
    
    if cat_detector is None or coco_classifier is None:
        st.error("❌ Failed to load models. Please check your model files!")
        st.info("💡 Make sure 'best.pt' is in the same directory as this script.")
        return
    
    st.success("✅ Models loaded successfully! Ready to detect Coco! 🎉")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## 🐱 About Coco Detection")
        st.markdown("""
        <div class="info-box">
            <strong>🔍 How it works:</strong><br>
            1. 📸 Upload your cat photo<br>
            2. 🤖 AI finds all cats<br>
            3. 🍊 Checks for orange fur<br>
            4. 🎯 Identifies if it's Coco!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>📋 Supported formats:</strong><br>
            • JPG, JPEG, PNG<br>
            • Any size (I'll optimize!)<br>
            • Clear photos work best! 📸
        </div>
        """, unsafe_allow_html=True)
    
    # Main upload area
    st.markdown("## 📸 Upload Your Cat Photo")
    
    uploaded_file = st.file_uploader(
        "Choose a cat image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of your cat. I work best with orange cats! 🧡"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Your Uploaded Photo")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to opencv format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with col2:
            st.markdown("### 🔍 Detection Results")
            
            if st.button("🚀 Start Coco Detection!", key="detect_button"):
                # Progress bar with messages
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                progress_messages = [
                    "🔍 Looking for cats...",
                    "🍊 Checking for orange fur...",
                    "🎯 Analyzing if it's Coco...",
                    "✨ Almost done..."
                ]
                
                for i, message in enumerate(progress_messages):
                    status_text.text(message)
                    progress_bar.progress((i + 1) * 25)
                    time.sleep(0.5)
                
                # Run detection
                try:
                    result = detect_coco_cat_streamlit(image_array, cat_detector, coco_classifier)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    if result["result"] == "is_coco":
                        st.markdown(f"""
                        <div class="success-msg">
                            🎉 HOORAY! This is our beloved Coco! 🧡<br>
                            Confidence: {result['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                        
                    elif result["result"] == "not_coco":
                        st.markdown(f"""
                        <div class="error-msg">
                            😸 This is a cute orange cat, but not our Coco!<br>
                            Confidence: {result['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.warning(f"😿 {result['message']}")
                    
                    # Show detailed info
                    with st.expander("📊 Detailed Detection Info"):
                        st.write(f"🐱 **Total cats found:** {result.get('total_cats', 0)}")
                        st.write(f"🍊 **Orange cats:** {result.get('orange_cats', 0)}")
                        st.write(f"🔧 **Detection method:** {result.get('detection_method', 'standard')}")
                        st.write(f"📈 **Confidence score:** {result['confidence']:.3f}")
                        if result.get('best_detection'):
                            st.write(f"🎯 **Best detection:** {result['best_detection']}")
                        
                        # Debug info for troubleshooting
                        st.write("🔍 **Debug Info:**")
                        st.write(f"- Model type: {'Classification' if 'Classification' in str(result.get('best_detection', '')) else 'Detection'}")
                        st.write(f"- Processing: Crops each orange cat individually")
                        st.write(f"- Detection mode: Standard (no fallback)")
                    
                except Exception as e:
                    st.error(f"❌ Oops! Something went wrong: {str(e)}")
                    st.info("💡 Try with a different image or check your model files.")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    <div class="cute-card">
        <h4 style="color: #ff9800; text-align: center;">💡 Pro Tips for Best Results</h4>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div style="text-align: center; margin: 1rem;">
                <h5>📸 Photo Quality</h5>
                <p>Clear, well-lit photos work best!</p>
            </div>
            <div style="text-align: center; margin: 1rem;">
                <h5>🐱 Cat Visibility</h5>
                <p>Make sure the cat is clearly visible!</p>
            </div>
            <div style="text-align: center; margin: 1rem;">
                <h5>🍊 Orange Cats</h5>
                <p>I'm specially trained for orange cats!</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
