import streamlit as st
import requests
import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import numpy as np
from openai import AzureOpenAI 
import os 
from dotenv import load_dotenv
load_dotenv()



# Azure API configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_API_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")

# Function to call Azure Document Intelligence Read API
def call_document_intelligence(file):
    # url = f"{AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}/formrecognizer/v2.1-preview.2/read/analyze"
    # headers = {
    #     "Ocp-Apim-Subscription-Key": AZURE_DOCUMENT_INTELLIGENCE_API_KEY,
    #     "Content-Type": "application/octet-stream"
    # }
    # params = {"readingOrder": "natural"}
    # response = requests.post(url, headers=headers, params=params, data=file)
    # response.raise_for_status()
    # operation_url = response.headers["Operation-Location"]
    
    
    # Poll the operation URL until the result is available
    while True:
        document_intelligence_client  = DocumentIntelligenceClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_API_KEY)
        )
        
        with open(file_path, "rb") as file:
            poller = document_intelligence_client.begin_analyze_document(
                model_id="prebuilt-read",  # Use prebuilt-read model
                body=file  # Pass the file content here
            )
        result = poller.result()
        if result:
            
            return result, result.content
        else:
            st.error("Document Intelligence processing failed.")
            return None

# Function to call Azure OpenAI GPT model
def call_azure_openai_gpt(content):
    client = AzureOpenAI(  
    azure_endpoint="https://iflocrpocgpt.openai.azure.com/",
    api_key="6jGu0ICdy50yiQ4VpHZZnxyXBlVggeQPljhtbwBbRHiKLIyuXlAlJQQJ99BAACHYHv6XJ3w3AAABACOGPEyT", 
    api_version="2024-05-01-preview",
    )
    # # Prepare the prompt for Azure OpenAI GPT model
    prompt = f"""
    The following is the content extracted from a resume:
    {content}

    Please extract the following details and return them as key-value pairs in JSON format:
    - Name
    - Years of Experience
    - Contact Number
    - Email
    - Skills
    """
    #Prepare the chat prompt 
    chat_prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
            "role":"user",
            "content":prompt
        }
    ] 
    
        
    # Include speech result if speech is enabled  
    messages = chat_prompt  
        
    # Generate the completion  
    completion = client.chat.completions.create(  
        model="gpt-35-turbo",
        messages=messages,
        max_tokens=800,  
        temperature=0.5,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False
    )
    
    
     # Extract the content from the ChatCompletion object
    content = completion.choices[0].message.content
    return content
def extract_json_from_gpt_output(gpt_output):
    try:
        # Parse the extracted content as JSON
        extracted_data = json.loads(gpt_output)
        return extracted_data
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse GPT output: {e}")
        return None

# Streamlit app
st.title("Resume Processing App")

# Form to accept user input
with st.form("resume_form"):
    # name = st.text_input("Name")
    # years_of_experience = st.number_input("Years of Experience", min_value=0, step=1)
    # contact_number = st.text_input("Contact Number")
    # email = st.text_input("Email")
    # skills = st.text_area("Skills")
    uploaded_file = st.file_uploader("Attach Resume File", type=["pdf", "docx", "txt","JPEG","PNG"])
    submitted = st.form_submit_button("Submit")

# Process the uploaded file
if submitted:
    if uploaded_file is not None:
        with st.spinner("Processing your file..."):
            # Call Azure Document Intelligence API
            # Save the uploaded file to the 'resumes' directory
            file_path = os.path.join("resumes", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            document_intelligence_result, content = call_document_intelligence(file_path)

            if document_intelligence_result:
                # # Extract the content from the Document Intelligence result

                
                extracted_data = call_azure_openai_gpt(content)
                extracted_data = extract_json_from_gpt_output(extracted_data)
                # # Parse GPT output and display it in Streamlit fields
                st.subheader("Extracted Information")
                try:
                    st.write(extracted_data)
                    # Extract and clean Years of Experience
                    years_of_experience_raw = extracted_data.get("Years of Experience", "0")
                    # Extract numeric value from the string
                    years_of_experience_cleaned = int(''.join(filter(str.isdigit, years_of_experience_raw)))

                    # Display fields
                    st.text_input("Name", value=extracted_data.get("Name", ""), key="name")
                    st.number_input(
                        "Years of Experience",
                        value=years_of_experience_cleaned,
                        key="experience",
                        min_value=0,
                    )
                    st.text_input("Contact Number", value=extracted_data.get("Contact Number", ""), key="contact_number")
                    st.text_input("Email", value=extracted_data.get("Email", ""), key="email")
                    st.text_area("Skills", value=extracted_data.get("Skills", ""), key="skills")
                except json.JSONDecodeError:
                    st.error("Failed to parse GPT output. Please check the response.")
    else:
        st.warning("Please upload a file before submitting.")

# import os
# import streamlit as st
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Azure API configuration
# AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
# AZURE_DOCUMENT_INTELLIGENCE_API_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

# # Ensure the 'resumes' directory exists
# if not os.path.exists("resumes"):
#     os.makedirs("resumes")

# # Function to call Azure Document Intelligence Read API
# def call_document_intelligence(file_path):
#     document_intelligence_client = DocumentIntelligenceClient(
#         endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
#         credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_API_KEY),
#     )

#     # Read the file content as bytes
#     with open(file_path, "rb") as file:
#         poller = document_intelligence_client.begin_analyze_document(
#             model_id="prebuilt-read",  # Use prebuilt-read model
#             body=file  # Pass the file content here
#         )
#     result = poller.result()
    
#     # Return the result and extracted content
#     if result:
#         return result, result.content
#     else:
#         st.error("Document Intelligence processing failed.")
#         return None

# # Streamlit app
# st.title("Resume Processing App")

# # File upload form
# uploaded_file = st.file_uploader("Attach Resume File", type=["pdf", "docx", "txt"])
# if st.button("Submit"):
#     if uploaded_file is not None:
#         with st.spinner("Processing your file..."):
#             # Save the uploaded file to the 'resumes' directory
#             file_path = os.path.join("resumes", uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             # Call Azure Document Intelligence API
#             document_intelligence_result, content = call_document_intelligence(file_path)

#             if document_intelligence_result:
#                 st.success("File processed successfully!")
#                 st.subheader("Extracted Content")
#                 st.write(content)
#     else:
#         st.warning("Please upload a file before submitting.")
