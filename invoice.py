from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Load the .env file to access API key
load_dotenv()

# Configure the API with your Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini Pro Vision model
model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

# Function to generate content using Gemini Pro Vision model
def get_gemini_response(input_text, image_data, user_prompt):
    try:
        # Generate content using the correct method
        response = model.generate_content([input_text, image_data[0], user_prompt])
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

# Function to handle image details
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Multilanguage Invoice Extractor")

st.header("Multilanguage Invoice Extractor")
input_text = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an Image of the invoice...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the invoice")

input_prompt = """
You are an expert in understanding invoices. We will upload an image as invoice
and you will have to answer any questions based on the uploaded invoice image
"""

# If the submit button is clicked:
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input_text)
    st.subheader("The Response is")
    st.write(response)
