<<<<<<< HEAD
# Resume_Parsing
=======
# Resume Parser with Gemini AI

This application uses Google's Gemini AI to parse and analyze resumes in PDF format.

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a Google Cloud account and get your Gemini API key

3. Add your Gemini API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- PDF resume upload and processing
- Extraction of key information including:
  - Personal Information
  - Professional Summary
  - Work Experience
  - Education
  - Skills
  - Certifications
  - Projects
- AI-powered analysis of resume strengths and improvements
- Download analysis results

## Requirements

See `requirements.txt` for a complete list of dependencies.
>>>>>>> ea97acf3 (Initial commit)
