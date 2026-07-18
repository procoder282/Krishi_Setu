import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Custom CSS for styling
st.markdown("""
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f0f7f4 0%, #e6f0ff 100%);
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    /* Constrain main content and add subtle card look */
    .main > div.block-container {
        max-width: 900px;
        margin: 1.25rem auto 3rem auto;
        padding: 1.25rem 1.5rem;
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    }

    /* Buttons */
    .stButton>button {
        color: white;
        background-color: #2b8aef;
        border-radius: 8px;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1f74d1;
        color: white;
    }

    /* File uploader button */
    .stFileUploader>div>div>div>button {
        color: white;
        background-color: #16a34a;
        border-radius: 8px;
    }

    /* Prediction result card */
    .prediction-result {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: linear-gradient(180deg, #ffffff 0%, #f7fffb 100%);
        margin: 1rem 0;
        border-left: 6px solid rgba(43,138,239,0.15);
    }

    /* Sidebar logo spacing */
    .sidebar-logo {
        display: block;
        margin: 0 auto 1.25rem auto;
        padding: 0.25rem;
        max-width: 220px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #0b2545;
    }

    /* Reduce expander font weight */
    .stExpanderHeader {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with Logo
st.sidebar.image("logo3.png",  use_container_width=True, caption="AI-Powered Crop Protection")

st.sidebar.title("AgriGuard AI")
app_mode = st.sidebar.radio("Navigate", ["Home", "About", "Crop Disease Recognition"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Upload a clear leaf image to get a fast AI diagnosis")

# Home Page
if app_mode == "Home":
    st.header("🌿 Smart Crop Disease Recognition")
    st.markdown("---")
    # Centered larger logo on Home
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("logo3.png", width=320, caption=None)

    # Two-column layout: image + brief intro
    left, right = st.columns([5,7])
    with left:
        image_path = "homeIMG.jpg"
        st.image(image_path, use_container_width=True, caption="Healthy crops start here")
    with right:
        st.subheader("Welcome to AgriGuard")
        st.markdown("<p>AgriGuard uses lightweight deep learning to detect common leaf diseases <strong style='color:#16a34a'>quickly</strong> and <strong style='color:#16a34a'>accurately</strong>.</p>", unsafe_allow_html=True)
        st.markdown("<p>Upload a clear photo of a single leaf and get an AI-powered diagnosis with suggested next steps.</p>", unsafe_allow_html=True)
        st.markdown("**How to use**")
        st.markdown("1. Capture a clear leaf photo.  2. Open *Crop Disease Recognition*.  3. Upload and run analysis.")
        st.markdown("**Why AgriGuard?**  <span style='color:#16a34a'>Fast results</span> • <span style='color:#16a34a'>Practical guidance</span> • Easy to use on mobile", unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.header("📚 About This Project")
    st.markdown("---")
    # Apply About-page specific green background and white text for readability
    st.markdown("""
        <style>
        .main > div.block-container {
            background: linear-gradient(135deg, #0f9d58 0%, #16a34a 100%) !important;
            color: #ffffff !important;
        }
        .main > div.block-container h1,
        .main > div.block-container h2,
        .main > div.block-container h3,
        .main > div.block-container p,
        .main > div.block-container a,
        .main > div.block-container li {
            color: #ffffff !important;
        }
        .main > div.block-container a { color: #dff7e0 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    with st.expander("🌐 The Project Overview", expanded=True):
        st.markdown("""
        This AI-powered solution helps farmers quickly identify plant diseases through leaf image analysis, 
        enabling early intervention and reducing crop losses.
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        #### Original Dataset
        - Source: [Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
        - Total Images: 87,000+ RGB images
        - Categories: 38 plant disease classes
        - Resolution: 256x256 pixels
        
        #### Our Implementation
        - Training Split: 70,295 images (80%)
        - Validation Split: 17,572 images (20%)
        - Test Set: 33 curated real-world images
        - Augmentation: Rotation, flipping, and zoom variations
        """)
    
    with st.expander("🛠️ Technical Architecture"):
        st.markdown("""
        - **Framework**: TensorFlow 2.0
        - **Model**: Custom CNN with 16-layer architecture
        - **Training**: 50 epochs with Adam optimizer
        - **Accuracy**: 98.7% validation accuracy
        - **Inference**: GPU-accelerated predictions
        """)
    st.write("© 2026 AgriGuard AI | Developed with care by the AgriGuard team")    
        

# Prediction Page
elif app_mode == "Crop Disease Recognition":
    st.header("Crop Disease Analysis 🔍")
    st.markdown("---")

    # File Upload Section
    st.subheader("Upload a leaf image")
    left_col, right_col = st.columns([6,4])
    with left_col:
        test_image = st.file_uploader("Choose a plant leaf image", type=["jpg", "png", "jpeg"], 
                                     help="Choose a clear photo of a single leaf")
    with right_col:
        st.markdown("<strong style='color:#16a34a'>Tips</strong>", unsafe_allow_html=True)
        st.markdown("<ul><li><span style='color:#ffffff00'> </span><strong style='color:#000000'>Use natural light</strong></li><li><strong style='color:#000000'>Include the whole leaf</strong></li><li><strong style='color:#000000'>Avoid blurred photos</strong></li></ul>", unsafe_allow_html=True)

    if test_image:
        # Image Preview
        st.subheader("Image Preview")
        st.image(test_image, use_container_width=True, caption="Uploaded Leaf Image")

        # Prediction Section
        st.subheader("Run Analysis")
        if st.button("Start Analysis 🚀", type="primary"):
            with st.spinner("Analyzing image, please wait..."):
                result_index = model_prediction(test_image)
                
                # Class Names Formatting
                class_name = [
                    'Apple - Apple Scab',
                    'Apple - Black Rot',
                    'Apple - Cedar Apple Rust',
                    'Apple - Healthy',
                    'Blueberry - Healthy',
                    'Cherry - Powdery Mildew',
                    'Cherry - Healthy',
                    'Corn - Cercospora Leaf Spot',
                    'Corn - Common Rust',
                    'Corn - Northern Leaf Blight',
                    'Corn - Healthy',
                    'Grape - Black Rot',
                    'Grape - Esca (Black Measles)',
                    'Grape - Leaf Blight',
                    'Grape - Healthy',
                    'Orange - Huanglongbing (Citrus Greening)',
                    'Peach - Bacterial Spot',
                    'Peach - Healthy',
                    'Bell Pepper - Bacterial Spot',
                    'Bell Pepper - Healthy',
                    'Potato - Early Blight',
                    'Potato - Late Blight',
                    'Potato - Healthy',
                    'Raspberry - Healthy',
                    'Soybean - Healthy',
                    'Squash - Powdery Mildew',
                    'Strawberry - Leaf Scorch',
                    'Strawberry - Healthy',
                    'Tomato - Bacterial Spot',
                    'Tomato - Early Blight',
                    'Tomato - Late Blight',
                    'Tomato - Leaf Mold',
                    'Tomato - Septoria Leaf Spot',
                    'Tomato - Spider Mites',
                    'Tomato - Target Spot',
                    'Tomato - Yellow Leaf Curl Virus',
                    'Tomato - Mosaic Virus',
                    'Tomato - Healthy'
                ]
                
                # Display Results
                st.markdown("---")
                st.subheader("Diagnosis Report")

                diagnosis = class_name[result_index]
                plant, disease = diagnosis.split(" - ")

                if "Healthy" in disease:
                    st.success(f"Great news — this {plant.lower()} appears healthy.")
                else:
                    st.error(f"Potential {disease} detected in {plant.lower()}. Consider treatment options below.")

                # Result Card
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="margin:0 0 .25rem 0; color:#000000;">Plant: {plant}</h3>
                    <h3 style="margin:0; color:#000000;">Condition: {disease}</h3>
                </div>
                """, unsafe_allow_html=True)
