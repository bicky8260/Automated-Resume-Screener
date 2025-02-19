'''PROJECT STRUCTURE'''
'''Automated_Resume_Screener/
├── app.py                  # Flask application
├── static/
│   ├── styles.css          # CSS styles            
├── templates/
│   ├── index.html          # Upload page
│   ├── results.html        # Results display page
│   ├── update.html         # Update skills page
├── uploaded_resumes/       # Folder to store uploaded resumes
├── resumes.db              # SQLite database
└── requirements.txt        # Python dependencies
'''
from flask import Flask,request,render_template,redirect,url_for
import sqlite3
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__) 

'''Folder to save uploaded resumes'''
UPLOAD_FOLDER = "uploaded_resumes"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

'''Database Setup'''
def create_database():
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS resumes (
                        id INTEGER PRIMARY KEY,
                        filename TEXT,
                        skills TEXT,
                        score REAL,
                        job_description TEXT
                      )''')
    conn.commit()
    conn.close()

def add_job_description_column():
    # Connect to your database
    conn = sqlite3.connect("resumes.db")  # Replace with your database file
    cursor = conn.cursor()

    # Check if the column already exists
    cursor.execute("PRAGMA table_info(resumes)")
    columns = [column[1] for column in cursor.fetchall()]  # Get all column names

    if "job_description" not in columns:
        # Add the column only if it does not exist
        cursor.execute("ALTER TABLE resumes ADD COLUMN job_description TEXT")
        conn.commit()
        print("Column 'job_description' added successfully.")
    else:
        print("Column 'job_description' already exists.")

    # Close the connection
    conn.close()

def insert_resume_data(filename, skills, score, job_description):
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM resumes WHERE filename = ?", (filename,))
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO resumes (filename, skills, score, job_description) VALUES (?, ?, ?, ?)", 
                   (filename, ", ".join(skills), score, job_description))
    else:
        cursor.execute("UPDATE resumes SET skills = ?, score = ?, job_description = ? WHERE filename = ?",
                   (", ".join(skills), score, job_description, filename))

    conn.commit()
    conn.close()

def fetch_ranked_resumes():
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, skills, score FROM resumes ORDER BY score DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

'''Text Extraction from PDFs'''
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

'''Information Extraction'''
def extract_skills_from_text(text, skill_keywords):
    skills = [keyword for keyword in skill_keywords if keyword.lower() in text.lower()]
    return skills

'''Job Description Analysis'''
import spacy
nlp = spacy.load("en_core_web_sm")
def process_job_description(job_description, skill_keywords):
    # Normalize job description: replace line breaks, commas, and extra spaces
    normalized_job_description = job_description.replace("\n", " ").replace(",", " ").lower()

    # Tokenize job description into words
    tokens = normalized_job_description.split()

    # Match tokens with skill keywords
    job_keywords = [token for token in tokens if token in [skill.lower() for skill in skill_keywords]]

    # Remove duplicates and return matched keywords
    return list(set(job_keywords))

'''Semantic Similarity Matching'''
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for semantic similarity

def calculate_semantic_score(resume_text, job_description):
    """
    Calculate the semantic similarity score between the resume text and the job description.
    """
    # Ensure neither input is empty
    if not resume_text or not job_description:
        print("Empty resume text or job description. Returning score 0.")
        return 0

    # Normalize and clean the text
    resume_text = " ".join(resume_text.lower().strip().split())  # Normalize whitespace
    job_description = " ".join(job_description.lower().strip().split())  # Normalize whitespace

    try:
        # Encode the texts using the model
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_description, convert_to_tensor=True)

        # Calculate cosine similarity (returns a value between 0 and 1)
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

        # Scale similarity score to a percentage (0-100)
        semantic_score = round(similarity_score * 100, 2)

        # Debug log: Print calculated similarity
        print(f"Semantic Similarity Score: {semantic_score}%")
        return semantic_score
    except Exception as e:
        # Handle unexpected errors
        print(f"Error calculating semantic score: {e}")
        return 0

'''Ranking'''
def rank_resumes(resumes, job_description, skill_keywords):
    # Extract keywords from the job description
    job_keywords = process_job_description(job_description, skill_keywords)
    print(f"Job Keywords Extracted: {job_keywords}")

    ranked_resumes = []
    for resume_name, data in resumes.items():
        # Normalize and extract resume skills
        resume_keywords = [skill.strip().lower() for skill in data["skills"]]
        print(f"Resume Keywords for {resume_name}: {resume_keywords}")

        # Calculate Keyword Score
        common_keywords = set(resume_keywords) & set(job_keywords)
        print(f"Common Keywords: {common_keywords}")
        keyword_score = len(common_keywords) / len(job_keywords) * 100 if job_keywords else 0
        print(f"Keyword Score: {keyword_score:.2f}%")

        # Calculate Semantic Similarity Score
        if data["text"].strip():
            semantic_score = calculate_semantic_score(data["text"], job_description)
        else:
            semantic_score = 0
        print(f"Semantic Score: {semantic_score:.2f}%")

        # Weighted Final Score
        final_score = (keyword_score * 0.8) + (semantic_score * 0.2)
        final_score_percentage = round(final_score, 2)
        print(f"Final Score: {final_score_percentage:.2f}%")

        ranked_resumes.append({"resume": resume_name, "score": f"{final_score_percentage}%"})

        # Insert Data into the Database
        insert_resume_data(resume_name, data["skills"], final_score_percentage)

    # Sort Resumes by Final Score
    ranked_resumes.sort(key=lambda x: float(x["score"].replace("%", "")), reverse=True)
    return ranked_resumes

'''Flask Routes'''
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    job_description = request.form.get("job_description")
    resumes = request.files.getlist("resumes")
    skill_keywords = [
    "Python", "Java", "Machine Learning", "Data Structures", "SQL",
    "NumPy", "Pandas", "Cloud", "Generative AI", "Object-Oriented Programming",
    "Deep Learning", "Algorithms", "Databases"
    ]

    resumes_data = {}
    for resume in resumes:
        resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        resume.save(resume_path)

        # Process resume
        text = extract_text_from_pdf(resume_path)
        skills = extract_skills_from_text(text, skill_keywords)
        resumes_data[resume.filename] = {"text": text, "skills": skills}

        # Calculate scores
        keyword_score = len(set([skill.lower() for skill in skills]) & set([keyword.lower() for keyword in job_description.split(",")])) / len(job_description.split(",")) * 100
        semantic_score = calculate_semantic_score(text, job_description)
        final_score = (keyword_score * 0.6) + (semantic_score * 0.4)

        # Save resume data and job description in the database
        insert_resume_data(resume.filename, skills, final_score, job_description)

    return redirect(url_for("results"))

@app.route("/delete/<int:resume_id>", methods=["POST"])
def delete_resume(resume_id):
    try:
        # Connect to the database
        conn = sqlite3.connect("resumes.db")
        cursor = conn.cursor()

        # Check if the resume exists and fetch its filename
        cursor.execute("SELECT filename FROM resumes WHERE id = ?", (resume_id,))
        row = cursor.fetchone()
        
        if row:
            filename = row[0]

            # Delete the resume entry from the database
            cursor.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
            conn.commit()

            # Delete the associated file if it exists
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Check if any rows are left in the database
        cursor.execute("SELECT COUNT(*) FROM resumes")
        row_count = cursor.fetchone()[0]

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        row_count = None  # Fallback if error occurs
    except Exception as e:
        print(f"Error: {e}")
        row_count = None
    finally:
        conn.close()

    # Redirect logic
    if row_count == 0:
        return redirect(url_for("index"))  # Redirect to the home page if no rows are left
    else:
        return redirect(url_for("results"))  # Redirect to the result page otherwise

@app.route("/update/<int:resume_id>", methods=["GET", "POST"])
def update_resume(resume_id):
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()

    if request.method == "POST":
        # Fetch the new inputs from the form
        new_job_description = request.form["job_description"].strip().lower()
        new_skills = [skill.strip().lower() for skill in request.form["skills"].split(",")]

        # Fetch the existing values from the database
        cursor.execute("SELECT job_description, skills FROM resumes WHERE id = ?", (resume_id,))
        row = cursor.fetchone()
        old_job_description = row[0].strip().lower()
        old_skills = [skill.strip().lower() for skill in row[1].split(",")]

        # Debug log: Print old and new values
        print(f"Old Job Description: {old_job_description}")
        print(f"New Job Description: {new_job_description}")
        print(f"Old Skills: {old_skills}")
        print(f"New Skills: {new_skills}")

        # Check if inputs have changed
        if new_job_description == old_job_description and set(new_skills) == set(old_skills):
            # Inputs are the same, skip recalculation and redirect to results
            print("No changes detected. Skipping update.")
            conn.close()
            return redirect(url_for("results"))

        # Fetch the resume text
        cursor.execute("SELECT filename FROM resumes WHERE id = ?", (resume_id,))
        row = cursor.fetchone()
        resume_text = ""
        if row:
            file_path = os.path.join(UPLOAD_FOLDER, row[0])
            if os.path.exists(file_path):
                resume_text = extract_text_from_pdf(file_path)

        # Normalize job keywords for proper comparison
        job_keywords = [keyword.strip().lower() for keyword in new_job_description.split(",") if keyword.strip()]

        # Debug log: Print extracted job keywords
        print(f"Job Keywords: {job_keywords}")

        # Recalculate scores only if job_keywords is not empty
        keyword_score = 0
        if job_keywords:
            keyword_score = (len(set(new_skills) & set(job_keywords)) / len(job_keywords)) * 100
        semantic_score = calculate_semantic_score(resume_text, new_job_description)

        # Combine the scores
        updated_score = round((keyword_score * 0.6) + (semantic_score * 0.4), 2)

        # Debug log: Print recalculated scores
        print(f"Keyword Score: {keyword_score}")
        print(f"Semantic Score: {semantic_score}")
        print(f"Final Updated Score: {updated_score}")

        # Update the database with the recalculated score
        cursor.execute(
            "UPDATE resumes SET job_description = ?, skills = ?, score = ? WHERE id = ?",
            (new_job_description, ", ".join(new_skills), updated_score, resume_id),
        )
        conn.commit()
        conn.close()

        return redirect(url_for("results"))

    else:
        # Fetch the current resume details to populate the form
        cursor.execute("SELECT id, filename, skills, score, job_description FROM resumes WHERE id = ?", (resume_id,))
        resume = cursor.fetchone()
        conn.close()
        return render_template("update.html", resume=resume)

@app.route("/results")
def results():
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()

    # Fetch the latest scores from the database
    cursor.execute("SELECT id, filename, skills, score FROM resumes ORDER BY score DESC")
    ranked_resumes = cursor.fetchall()
    conn.close()

    return render_template("results.html", resumes=ranked_resumes)

if __name__ == "__main__":
    create_database()  # Ensure the database is created before starting the app
    add_job_description_column()  # Add the column if it doesn't exist
    app.run(debug=True, port=8080)
