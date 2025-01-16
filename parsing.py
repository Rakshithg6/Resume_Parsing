import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def calculate_skill_match_score(resume_text, required_skills):
    # Convert to lowercase for better matching
    resume_text = resume_text.lower()
    required_skills = [skill.lower() for skill in required_skills]
    
    # Calculate matches
    matched_skills = [skill for skill in required_skills if skill in resume_text]
    match_score = len(matched_skills) / len(required_skills) * 100
    
    return match_score, matched_skills

def extract_technical_tools(text):
    # Common technical tools and frameworks
    tools_patterns = {
        'Data Science': r'(python|r|matlab|jupyter|pandas|numpy|scipy|scikit-learn|tensorflow|keras|pytorch|tableau|power bi)',
        'Database': r'(sql|mysql|postgresql|mongodb|oracle|cassandra|redis)',
        'Big Data': r'(hadoop|spark|hive|pig|kafka|airflow)',
        'Cloud': r'(aws|azure|gcp|google cloud|cloud computing)',
        'Version Control': r'(git|github|gitlab|bitbucket)',
        'IDE/Tools': r'(vscode|pycharm|jupyter|spyder|rstudio)',
        'Preprocessing': r'(pandas|numpy|data cleaning|feature engineering|data transformation|data preprocessing)'
    }
    
    found_tools = {}
    text = text.lower()
    
    for category, pattern in tools_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            found_tools[category] = list(set(matches))
    
    return found_tools

def parse_resume_for_hr(text, required_skills):
    # Initialize Gemini Pro model
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create prompt template for HR-specific analysis
    prompt = PromptTemplate(
        input_variables=["resume_text", "required_skills"],
        template="""Please analyze the following resume focusing on skills, projects, and certifications for HR evaluation:
        
        Resume Text: {resume_text}
        Required Skills: {required_skills}
        
        Please provide the following information in a structured format:
        
        1. Skills Analysis:
           - Technical Skills
           - Soft Skills
           - Skill Match with Requirements
        
        2. Projects:
           - List relevant projects
           - Technologies used
           - Project impact/results
        
        3. Certifications:
           - List all certifications
           - Relevance to required skills
        
        4. Tools & Technologies:
           - List all technical tools mentioned
           - Proficiency indicators (if available)
        
        5. Candidate Evaluation:
           - Overall match score
           - Strengths
           - Areas for improvement
           - Recommendation for role fit
        """
    )
    
    # Create and run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(resume_text=text, required_skills=", ".join(required_skills))
    return response

def main():
    st.set_page_config(page_title="HR Resume Parser", layout="wide")
    
    st.title("🎯 Smart Resume Parser for HR")
    st.write("Upload resumes and match candidates based on skills and requirements")
    
    # Input for required skills
    st.sidebar.header("Job Requirements")
    job_role = st.sidebar.text_input("Job Role (e.g., Data Scientist, Software Engineer)")
    required_skills = st.sidebar.text_area("Required Skills (one per line)")
    required_skills_list = [skill.strip() for skill in required_skills.split("\n") if skill.strip()]
    
    uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")
    
    if uploaded_file is not None and required_skills_list:
        with st.spinner("Analyzing resume..."):
            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                # Calculate skill match score
                match_score, matched_skills = calculate_skill_match_score(resume_text, required_skills_list)
                
                # Extract technical tools
                tools_found = extract_technical_tools(resume_text)
                
                # Parse resume using Gemini
                analysis = parse_resume_for_hr(resume_text, required_skills_list)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"Match Score: {match_score:.1f}%")
                    st.write("### Matched Skills")
                    for skill in matched_skills:
                        st.write(f"✅ {skill}")
                
                with col2:
                    st.write("### Technical Tools Detected")
                    for category, tools in tools_found.items():
                        st.write(f"**{category}:** {', '.join(tools)}")
                
                st.markdown("### Detailed Analysis")
                st.write(analysis)
                
                # Add download button for the analysis
                st.download_button(
                    label="Download Analysis Report",
                    data=f"""Resume Analysis Report
                    
Match Score: {match_score:.1f}%

Matched Skills:
{chr(10).join(['- ' + skill for skill in matched_skills])}

Technical Tools:
{chr(10).join([f'{category}: {", ".join(tools)}' for category, tools in tools_found.items()])}

Detailed Analysis:
{analysis}
                    """,
                    file_name="resume_analysis.txt",
                    mime="text/plain"
                )
    
    elif not required_skills_list and uploaded_file:
        st.warning("Please enter required skills in the sidebar.")

if __name__ == "__main__":
    main()