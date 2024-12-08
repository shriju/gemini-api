# import google.generativeai as genai
# import os

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# try:
#     models = genai.list_models()
#     print("Available models:")
#     for model in models:
#         print(model)
# except Exception as e:
#     print("Error:", e)

# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# print(f"GOOGLE_API_KEY: {api_key}")

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Explicitly configure with the API key
genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    print("Available models:")
    for model in models:
        print(model)
except Exception as e:
    print("Error:", e)