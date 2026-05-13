try:
    import streamlit as st
except Exception:
    st = None

try:
    import google.generativeai as genai
except Exception:
    genai = None
import re
try:
    from PIL import Image
except Exception:
    Image = None
import io
try:
    from fpdf import FPDF
except Exception:
    FPDF = None
import os
import json
from datetime import datetime
import uuid

# ================= CONFIGURATION =================
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# If running inside Streamlit, ensure API key is present and configure Gemini
if st is not None:
    if "GOOGLE_API_KEY" not in getattr(st, "secrets", {}):
        st.error("API Key not found. Please set GOOGLE_API_KEY in Streamlit secrets.")
        st.stop()
    if genai is not None:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# ================= SYSTEM PROMPT =================
system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images.

Your responsibilities include:

1. Detailed Analysis
2. Findings Report
3. Recommendations and Next Steps
4. Treatment Suggestions

5. Confidence Score:
Always include a confidence score explicitly in this format:
"Confidence Score: XX%"

Important Notes:
- Only respond if image is related to human health
- If unclear, say "Unable to determine"
- Always include this disclaimer:
"Consult with a Doctor before making any decisions"
"""

# ================= CONFIDENCE EXTRACTOR =================
def extract_confidence(text):
    """Extract a confidence value from model text.

    Handles common variants such as:
    - 'Confidence Score: 87%'
    - 'confidence: 0.87' (interpreted as fraction)
    - '87 percent' or '87%'
    - 'Confidence is 87.5%'

    Returns a float in range 0..100 or None if not found.
    """
    if not text:
        return None

    # 1) Percent with explicit '%' sign (e.g. '87%', '100%')
    m = re.search(r"(?P<num>\d{1,3}(?:\.\d+)?)\s*%", text)
    if m:
        try:
            val = float(m.group('num'))
            return max(0.0, min(100.0, val))
        except Exception:
            pass

    # 2) Keywords with 'percent' (e.g. '87 percent')
    m = re.search(r"(?i)(?P<num>\d{1,3}(?:\.\d+)?)\s*(?:percent|per cent)\b", text)
    if m:
        try:
            val = float(m.group('num'))
            return max(0.0, min(100.0, val))
        except Exception:
            pass

    # 3) Confidence near a fractional value (e.g. 'confidence: 0.87', '.87')
    m = re.search(r"(?i)confidence[^0-9\n\r]{0,10}(?P<num>0?\.\d+)", text)
    if m:
        try:
            val = float(m.group('num')) * 100.0
            return max(0.0, min(100.0, val))
        except Exception:
            pass

    # 4) Generic standalone fraction like '0.87' (prefer only if 'confidence' nearby)
    m = re.search(r"(?i)(?:confidence).{0,40}?([0]?\.?\d{1,3})", text)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val * 100.0
            if 0.0 <= val <= 100.0:
                return val
        except Exception:
            pass

    return None

# ================= PDF GENERATOR =================
def generate_pdf(clean_text, confidence):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    try:
        pdf.image("./logo.jpeg", x=10, y=8, w=33)
    except:
        pass
        
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(30, 144, 255) # DodgerBlue
    pdf.cell(0, 20, "Medical Analysis Report", ln=True, align="R")
    
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Content
    if confidence is not None:
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Confidence Score: {confidence:.2f}%", ln=True)
        pdf.ln(5)
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(50, 50, 50)
    # Basic cleaning for standard PDF fonts
    safe_text = clean_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 8, safe_text)
    
    # Footer
    pdf.ln(20)
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 10, "Disclaimer: This report is AI-generated and intended for informational purposes only. Consult with a qualified healthcare professional before making any medical decisions.", align="C")
    
    return pdf.output()

# ================= HISTORY MANAGEMENT =================
HISTORY_DIR = "history"
METADATA_FILE = os.path.join(HISTORY_DIR, "metadata.json")
IMAGES_DIR = os.path.join(HISTORY_DIR, "images")

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR, exist_ok=True)

def save_to_history(image, confidence, clean_text, filename):
    history = []
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_id = str(uuid.uuid4())
    image_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    image.save(image_path)
    
    entry = {
        "id": image_id,
        "timestamp": timestamp,
        "image_name": filename,
        "image_path": image_path,
        "confidence": confidence,
        "text": clean_text
    }
    
    history.insert(0, entry) # Most recent first
    with open(METADATA_FILE, "w") as f:
        json.dump(history, f, indent=4)
    return entry

def load_history():
    if not os.path.exists(METADATA_FILE):
        return []
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except:
        return []

# Only run Streamlit UI / API code when Streamlit is available
if st is not None:
    # ================= STREAMLIT CONFIG =================
    st.set_page_config(page_title="Disease Identifier", page_icon="🧑‍⚕️")

    # Theme state
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = True

    col1, col2 = st.columns([8, 2])
    with col2:
        st.toggle("Dark Mode", key="theme_mode")

    # Theme colors
    theme_colors = {
        True: {'bg': '#1E1E1E', 'text': '#FFFFFF'},
        False: {'bg': '#FFFFFF', 'text': '#000000'}
    }[st.session_state.theme_mode]

    # Apply CSS
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {theme_colors['bg']};
            color: {theme_colors['text']};
        }}
    </style>
    """, unsafe_allow_html=True)

    # ================= UI =================
    st.image("./logo.jpeg", width=200)
    st.title("Disease Identifier 🧑‍⚕️")
    st.header("Upload medical images to analyze diseases and get AI insights")

    upload_files = st.file_uploader(
        "Upload images",
        type=["jpeg", "jpg", "png"],
        accept_multiple_files=True
    )

    # File size validation (5MB limit)
    if upload_files:
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
        valid_files = []
        
        for file in upload_files:
            if file.size > MAX_FILE_SIZE:
                st.warning(f"⚠️ File '{file.name}' exceeds 5MB limit and will be skipped.")
            else:
                valid_files.append(file)
        
        if not valid_files:
            st.error("❌ No valid files to process. All files exceed 5MB limit.")
            st.stop()
        
        upload_files = valid_files

    # Preview uploaded images
    if upload_files:
        for i, file in enumerate(upload_files):
            st.image(file, width=200, caption=f"Image {i+1}")

    submit_button = st.button("Generate Analysis")

    # ================= MAIN LOGIC =================
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "view_history" not in st.session_state:
        st.session_state.view_history = None

    # Sidebar History
    with st.sidebar:
        st.markdown("---")
        st.header("📜 Analysis History")
        history = load_history()
        if not history:
            st.info("No past analyses found.")
        else:
            if st.button("Clear History"):
                if os.path.exists(METADATA_FILE):
                    os.remove(METADATA_FILE)
                # Optionally delete images too, but for safety let's just clear metadata
                st.rerun()
                
            for item in history:
                # Display history items as buttons
                if st.button(f"{item['timestamp']}\n{item['image_name']}", key=item['id'], use_container_width=True):
                    st.session_state.view_history = item

            if st.session_state.view_history:
                if st.button("⬅️ Back to Current", use_container_width=True):
                    st.session_state.view_history = None
                    st.rerun()

    if submit_button:
        st.session_state.view_history = None # Clear history view when new analysis is triggered
        if not upload_files:
            st.error("Please upload at least one image before clicking 'Generate Analysis'.")
        else:
            st.session_state.analysis_results = [] # Reset for new batch
            for i, file in enumerate(upload_files):
                # ===== IMAGE PROCESSING + ERROR HANDLING =====
                try:
                    image_data = file.getvalue()
                    image = Image.open(io.BytesIO(image_data))
                except Exception:
                    st.error(f"Error processing image {i+1}")
                    continue

                # ===== API CALL =====
                with st.spinner(f"Analyzing Image {i+1}..."):
                    try:
                        if genai is None:
                            raise RuntimeError("Generative API not available")
                        model = genai.GenerativeModel('models/gemini-2.5-flash')
                        response = model.generate_content(
                            [system_prompt, image],
                            generation_config=generation_config
                        )
                    except Exception as e:
                        st.error(f"API Error for Image {i+1}: {str(e)}")
                        continue

                # ===== RESPONSE VALIDATION =====
                if not response or not getattr(response, "text", None):
                    st.error(f"Invalid response for Image {i+1}")
                    continue

                # ===== CONFIDENCE EXTRACTION =====
                confidence = extract_confidence(response.text)

                # ===== CLEAN RESPONSE =====
                clean_text = response.text or ""

                # Remove any lines that mention confidence (case-insensitive)
                clean_text = re.sub(r'(?im)^[ \t]*.*\bconfidence\b.*$', '', clean_text).strip()

                # Remove explicit percent tokens (e.g. '87%')
                clean_text = re.sub(r'(?i)\b\d{1,3}(?:\.\d+)?\s*%\b', '', clean_text)

                # Remove 'percent' spelled-out forms (e.g. '87 percent')
                clean_text = re.sub(r'(?i)\b\d{1,3}(?:\.\d+)?\s*(?:percent|per cent)\b', '', clean_text)

                # Remove fractional confidence mentions near the word 'confidence' (e.g. 'confidence: 0.87')
                clean_text = re.sub(r'(?i)confidence[^\n]{0,40}?0?\.\d+', '', clean_text)

                # Collapse excessive blank lines
                clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text).strip()

                st.session_state.analysis_results.append({
                    "image": image,
                    "confidence": confidence,
                    "clean_text": clean_text,
                    "original_index": i + 1
                })
                save_to_history(image, confidence, clean_text, file.name)

    # Display Persisted Results
    results_to_show = []
    if st.session_state.view_history:
        h = st.session_state.view_history
        try:
            results_to_show = [{
                "image": Image.open(h['image_path']),
                "confidence": h['confidence'],
                "clean_text": h['text'],
                "title": f"Historical Report: {h['image_name']}",
                "timestamp": h['timestamp']
            }]
        except Exception as e:
            st.error(f"Error loading history: {e}")
    else:
        results_to_show = [
            {**r, "title": f"Analysis for Image {r['original_index']}", "timestamp": None} 
            for r in st.session_state.analysis_results
        ]

    for result in results_to_show:
        st.markdown("---")
        if result.get("timestamp"):
            st.caption(f"Analysis from {result['timestamp']}")
        st.title(result['title'])
        st.image(result['image'], width=300)

        if result['confidence'] is not None:
            st.progress(result['confidence'] / 100)
            st.markdown(f"### Confidence Score: **{result['confidence']:.2f}%**")

            if result['confidence'] > 80:
                st.success("High Confidence Prediction")
            elif result['confidence'] > 50:
                st.info("Moderate Confidence Prediction")
            else:
                st.error("Low Confidence Prediction")
        else:
            st.warning("⚠️ Confidence score not found in AI response")

        st.write(result['clean_text'])

        # ===== PDF DOWNLOAD BUTTON =====
        pdf_bytes = generate_pdf(result['clean_text'], result['confidence'])
        st.download_button(
            label="📥 Download Analysis Report as PDF",
            data=bytes(pdf_bytes),
            file_name=f"analysis_report_{result.get('original_index', 'history')}.pdf",
            mime="application/pdf"
        )

        st.warning("⚠️ Disclaimer: Consult with a Doctor before making any decisions")
