# Automated-Resume-Screener
A Python-based resume screening tool that automates the process of extracting and analyzing key information from resumes. It efficiently filters candidates based on predefined criteria, helping recruiters streamline the hiring process.
Features:
✅ Extracts candidate details (name, contact, skills, experience) from resumes
✅ Supports multiple file formats (PDF, DOCX)
✅ Keyword-based skill matching with job descriptions
✅ Semantic similarity analysis for better ranking
✅ Stores resumes and rankings in an SQLite database
✅ Web-based interface for easy interaction
✅ Allows updating and deleting resume records

Technologies Used:
1.Backend: Python, Flask, SQLite
2.Frontend: HTML, CSS, JavaScript
3.Resume Parsing: pdfminer, python-docx
4.Natural Language Processing: spaCy, NLTK, SentenceTransformers
5.Semantic Matching: all-MiniLM-L6-v2 model for similarity scoring
