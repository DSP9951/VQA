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
        background-image: url("https://img.freepik.com/free-photo/modern-background-with-lines_1361-3533.jpg");
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

    /* Style the chatbot interface */
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .chat-message {
        margin-bottom: 15px;
    }

    .chat-message.user {
        text-align: right;
    }

    .chat-message.bot {
        text-align: left;
    }

    .chat-message p {
        display: inline-block;
        padding: 10px 15px;
        border-radius: 10px;
        max-width: 70%;
    }

    .chat-message.user p {
        background-color: #0078d4;
        color: white;
    }

    .chat-message.bot p {
        background-color: #e1e1e1;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title of the app with custom styling
st.markdown('<h1 class="title">Visual Question Answering with Chatbot</h1>', unsafe_allow_html=True)

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

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.markdown('<h3 style="color: #8C564B;">Chat with the Bot:</h3>', unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot"><p>{message["content"]}</p></div>', unsafe_allow_html=True)

    # Ask a question about the image
    user_question = st.text_input("Enter your question here", key="question")

    # Process the image and generate an answer
    if st.button("Send", key="send_button"):
        if user_question.strip() == "":
            st.error("Please enter a question.")
        else:
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Preprocess the image and question
            inputs = processor(image, user_question, return_tensors="pt")

            # Generate an answer using the BLIP model
            with st.spinner("Processing image and generating an answer..."):
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)

                # Add bot answer to chat history
                st.session_state.chat_history.append({"role": "bot", "content": answer})
                import streamlit as st

def handle_click():
    st.session_state.clicked = True



if "clicked" in st.session_state and st.session_state.clicked:
    st.write("Hello")
    del st.session_state.clicked  # Clean up the state
            
            # Rerun the app to update the chat history display
            #st.experimental_rerun()
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
