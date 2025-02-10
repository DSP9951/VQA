# Import required libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Custom CSS to set an image as the background
st.markdown(
    """
    <style>
    /* Set an image as the background */
    .stApp {
        background-image: url("data:image/jpeg;base64,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");
        background-size: cover;  /* Cover the entire app */
        background-position: center;  /* Center the image */
        background-repeat: no-repeat;  /* Prevent repeating */
        background-attachment: fixed;  /* Fix the background while scrolling */
    }

    /* Add a semi-transparent overlay to improve readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.7);  /* White overlay with 70% opacity */
        z-index: -1;  /* Place the overlay behind the content */
    }

    /* Style the title */
    .title {
        text-align: center;
        color: #FF4B4B;
        font-family: "Helvetica Neue", sans-serif;
        font-size: 2.5em;
        margin-bottom: 20px;
    }

    /* Style headers */
    h3 {
        color: #1F77B4;
        font-family: "Helvetica Neue", sans-serif;
    }

    /* Style buttons */
    .stButton button {
        background-color: #11e721;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1em;
        font-family: "Helvetica Neue", sans-serif;
    }

    /* Style the footer */
    .footer {
        text-align: center;
        color: #7F7F7F;
        font-family: "Helvetica Neue", sans-serif;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title of the app with custom styling
st.markdown('<h1 class="title">Visual Question Answering</h1>', unsafe_allow_html=True)

# Function to load the BLIP model and processor
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

# Load the BLIP model and processor
processor, model = load_model()

# Let the user choose between uploading an image or capturing from the camera
st.markdown('<h3>Choose an option to provide an image:</h3>', unsafe_allow_html=True)
option = st.radio("Select an option:", ("Upload an image", "Capture from camera"), key="option")

# Initialize the image variable
image = None

# Handle image upload
if option == "Upload an image":
    st.markdown('<h3 style="color: #2CA02C;">Upload an image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Handle camera capture
elif option == "Capture from camera":
    st.markdown('<h3 style="color: #D62728;">Capture an image from your camera</h3>', unsafe_allow_html=True)
    camera_image = st.camera_input("Take a picture", key="camera")
    if camera_image is not None:
        image = Image.open(camera_image)

# Display the image if available
if image is not None:
    st.markdown('<h3 style="color: #9467BD;">Image:</h3>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Ask a question about the image
    st.markdown('<h3 style="color: #8C564B;">Ask a question about the image:</h3>', unsafe_allow_html=True)
    question = st.text_input("Enter your question here", key="question")

    # Process the image and generate an answer
    if st.button("Get Answer", key="answer_button"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            # Preprocess the image and question
            inputs = processor(image, question, return_tensors="pt")

            # Generate an answer using the BLIP model
            st.markdown('<h3 style="color: #E377C2;">Answer:</h3>', unsafe_allow_html=True)
            with st.spinner("Processing image and generating an answer..."):
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"*Answer:* {answer}")
else:
    st.warning("Please upload an image or capture one from the camera.")

# Add a colorful footer
st.markdown(
    """
    <div class="footer">
        <hr>
        <p>Powered by Streamlit and BLIP 🤖</p>
    </div>
    """,
    unsafe_allow_html=True
)
