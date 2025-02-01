import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(uploaded_file):
    try:
        # Read PDF file
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        
        # Extract text from each page
        for page in pdf_document:
            text += page.get_text()
            
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def parse_resume(text):
    # Initialize Gemini Pro model
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""Please analyze the following resume and extract key information in a structured format:
        
        Resume Text: {resume_text}
        
        Please provide the following information:
        1. Personal Information (Name, Contact Details)
        2. Professional Summary
        3. Work Experience (with dates and key responsibilities)
        4. Education
        5. Skills
        6. Certifications (if any)
        7. Projects (if any)
        
        Also provide a brief assessment of the resume's strengths and areas for improvement."""
    )
    
    # Create and run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(resume_text=text)
    return response

def main():
    st.set_page_config(page_title="Resume Parser", layout="wide")
    
    st.title("📄 Resume Parser with Gemini AI")
    st.write("Upload a resume in PDF format to analyze it using AI")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing resume..."):
            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                # Parse resume using Gemini
                analysis = parse_resume(resume_text)
                
                # Display results
                st.success("Resume Analysis Complete!")
                st.markdown("### Analysis Results")
                st.write(analysis)
                
                # Add download button for the analysis
                st.download_button(
                    label="Download Analysis",
                    data=analysis,
                    file_name="resume_analysis.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
