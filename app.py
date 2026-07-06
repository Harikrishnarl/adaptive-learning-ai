from flask import Flask, render_template, request, jsonify, session
import requests
import os
import json
import uuid
import hashlib
from dotenv import load_dotenv
from datetime import datetime
import math
import PyPDF2
from io import BytesIO

# ─── LangChain Integrations ──────────────────────────────────────────────────
# ─── NEW CLEAN IMPORTS (LangChain v0.2+) ───────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama  # 👈 Cleaned and moved here
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "adaptive-learning-secret-2024")

# ─── API & Core Config ───────────────────────────────────────────────────────
MB_API_KEY = os.getenv("MB_API_KEY")
MB_BASE    = "https://mem-brain-api-cutover-v4-production.up.railway.app/api/v1"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")  # Inference Model
EMBED_MODEL = "nomic-embed-text"                  # Embedding Model

MB_HEADERS = {
    "Authorization": f"Bearer {MB_API_KEY}",
    "Content-Type":  "application/json"
}

# Initialize LangChain Core Components
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL, 
    model=LLAMA_MODEL, 
    temperature=0.2,
    format="json"  # Forces native structured outputs
)

# Global runtime dictionary to keep LangChain FAISS indexes in RAM per user/subject
VECTOR_REGISTRY = {}

# ─── LOCAL DB ─────────────────────────────────────────────────────────────────
DB_FILE = "hackathon_db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump({"curriculums": {}, "notes": {}, "performance": {}, "users": {}}, f)
    try:
        with open(DB_FILE, 'r') as f:
            db = json.load(f)
            if "users" not in db: db["users"] = {}
            return db
    except:
        return {"curriculums": {}, "notes": {}, "performance": {}, "users": {}}

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

# ─── MemBrain Logging ────────────────────────────────────────────────────────
def mb_store(content: str, tags: list):
    try:
        requests.post(f"{MB_BASE}/memories", json={"content": content, "tags": tags}, headers=MB_HEADERS, timeout=3)
    except: pass

# ─── Student Identity & Retention Analytics ──────────────────────────────────
def make_student_id(email: str) -> str:
    h = hashlib.md5(email.strip().lower().encode()).hexdigest()[:8]
    clean = email.strip().lower().split('@')[0].replace(".", "_")
    return f"stu_{clean}_{h}"

def get_student_id() -> str:
    if "student_id" not in session:
        session["student_id"] = f"anon_{str(uuid.uuid4())[:8]}"
    return session["student_id"]

def get_student_name() -> str:
    return session.get("student_name", "Student")

def get_user_curriculum(student_id: str) -> dict:
    return load_db()["curriculums"].get(student_id, {})

def save_user_curriculum(student_id: str, curriculum: dict):
    db = load_db()
    db["curriculums"][student_id] = curriculum
    save_db(db)

def record_quiz_result(student_id: str, subject: str, topic: str, score: int, total: int):
    db = load_db()
    if student_id not in db["performance"]: db["performance"][student_id] = {}
    if subject not in db["performance"][student_id]: db["performance"][student_id][subject] = {}
    if topic not in db["performance"][student_id][subject]: db["performance"][student_id][subject][topic] = []
    
    pct = round((score / total) * 100, 1)
    db["performance"][student_id][subject][topic].append({"pct": pct, "date": datetime.now().isoformat()})
    save_db(db)
    mb_store(f"Scored {pct}% on {topic} in {subject}", ["quiz", subject, topic, student_id])

def get_student_performance(student_id: str) -> dict:
    return load_db()["performance"].get(student_id, {})

def calculate_retention(attempts_with_dates: list) -> dict:
    if not attempts_with_dates:
        return {"R": 0.0, "t": 999, "S": 1, "status": "never_studied", "next_review_days": 0, "last_score": 0, "attempts": 0}
    sorted_attempts = sorted(attempts_with_dates, key=lambda x: x["date"])
    last = sorted_attempts[-1]
    try:
        last_date = datetime.fromisoformat(last["date"].replace("Z", "").split("+")[0])
        t = max(0, (datetime.now() - last_date).total_seconds() / 86400)
    except: t = 1.0

    S = 1.0
    for attempt in sorted_attempts:
        w = attempt["pct"] / 100.0
        S += (3.0 * w) if w >= 0.75 else (1.5 * w) if w >= 0.50 else (0.5 * w)
    S = min(S, 21.0)
    R = round(math.exp(-t / S), 3)

    if R >= 0.80: status, next_review_days = "safe", int(S * 0.7)
    elif R >= 0.65: status, next_review_days = "good", 2
    elif R >= 0.40: status, next_review_days = "review_due", 0
    else: status, next_review_days = "critical", 0

    return {"R": R, "t": round(t, 1), "S": round(S, 1), "status": status, "next_review_days": next_review_days, "last_score": last["pct"], "attempts": len(sorted_attempts)}

def compute_analytics(performance: dict) -> dict:
    strengths, weaknesses, topic_avgs = [], [], {}
    all_memory_strengths = [] 
    
    for subj, topics in performance.items():
        for topic, attempts in topics.items():
            scores = [a["pct"] for a in attempts]
            avg = sum(scores) / len(scores)
            ri = calculate_retention(attempts)
            all_memory_strengths.append(ri["S"])
            
            topic_avgs[f"{subj}||{topic}"] = {
                "avg": round(avg, 1), "subject": subj, "topic": topic, "attempts": len(scores),
                "retention": ri["R"], "retention_status": ri["status"], "memory_strength": ri["S"]
            }
            entry = {"label": topic, "subject": subj, "score": round(avg, 1), "retention": ri["R"], "retention_status": ri["status"]}
            (strengths if avg >= 75 else weaknesses).append(entry)

    strengths.sort(key=lambda x: -x["score"])
    weaknesses.sort(key=lambda x: x["score"])
    
    subject_scores = {s: round(sum(a["pct"] for ts in t.values() for a in ts) / sum(len(ts) for ts in t.values()), 1) for s, t in performance.items() if t}
    forgetting_data = [
        {"topic": v["topic"], "subject": v["subject"], "retention": round(v["retention"] * 100, 1), "priority": "HIGH" if v["retention"] < 0.70 else "MEDIUM"}
        for v in topic_avgs.values() 
    ]
    forgetting_data.sort(key=lambda x: x["retention"])
    forgetting_data = forgetting_data[:8]
    avg_s = sum(all_memory_strengths) / len(all_memory_strengths) if all_memory_strengths else 1.0
    
    return {
        "strengths": strengths[:6], "weaknesses": weaknesses[:6], "topic_avgs": topic_avgs, 
        "subject_scores": subject_scores, "forgetting": forgetting_data, 
        "overall_avg": round(sum(v["avg"] for v in topic_avgs.values()) / len(topic_avgs), 1) if topic_avgs else 0, 
        "total_attempts": sum(v["attempts"] for v in topic_avgs.values()), "avg_memory_strength": round(avg_s, 2) 
    }

# ─── LANGCHAIN RAG INFERENCE GENERATION ───────────────────────────────────────

def generate_questions(subject: str, topic: str, difficulty: str, style: str, student_id: str) -> list:
    rag_key = f"{student_id}||{subject}"
    context_text = ""
    
    # Check if a LangChain FAISS index exists for this subject
    if rag_key in VECTOR_REGISTRY:
        print(f"🔍 LangChain Vector Retrieval: Searching for top matching chunks for '{topic}'...")
        vector_store = VECTOR_REGISTRY[rag_key]
        
        # Retrieve the top 3 most semantically aligned documents
        matched_docs = vector_store.similarity_search(topic, k=3)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        
    if not context_text.strip():
        context_text = "Standard foundational syllabus criteria and educational core metrics."

    style_desc = {
        "visual": "spatial reasoning diagrams", "analytical": "logical progressions", 
        "practical": "real-world application cases", "conceptual": "definitions and high-level structural theory"
    }.get(style, "balanced distribution")

    # Define LangChain Prompt Template
    template = """You are an expert AI tutor. Generate exactly 5 multiple-choice quiz questions for the topic "{topic}" in the subject "{subject}".
Difficulty: {difficulty}
Learning Style: {style_desc}

IMPORTANT INSTRUCTIONS:
1. Base the questions strictly on this relevant context extracted from the user's files:
--- RELEVANT CONTEXT CHUNKS ---
{context_text}
--- END RELEVANT CONTEXT ---

Return a JSON array containing exactly 5 question objects. 
Use this EXACT JSON schema:
[
  {{"question": "string", "options": ["A) string", "B) string", "C) string", "D) string"], "correct": 0, "explanation": "string"}}
]"""

    prompt = PromptTemplate(
        input_variables=["topic", "subject", "difficulty", "style_desc", "context_text"],
        template=template
    )

    # Set up the LangChain Run Execution Chain (LCEL)
    chain = prompt | llm | JsonOutputParser()

    try:
        # Invoke the LangChain Chain
        parsed_json = chain.invoke({
            "topic": topic,
            "subject": subject,
            "difficulty": difficulty,
            "style_desc": style_desc,
            "context_text": context_text
        })
        
        if isinstance(parsed_json, dict) and "questions" in parsed_json:
            parsed_json = parsed_json["questions"]

        if isinstance(parsed_json, list) and len(parsed_json) > 0:
            return parsed_json
        else:
            raise ValueError("LangChain execution chain output was empty or invalid.")
            
    except Exception as e:
        print(f"\n❌ LANGCHAIN QUIZ GEN ERROR: {str(e)}\n")
        return [
            {
                "question": f"Error running LangChain quiz generation chain.", 
                "options": ["A) Inspect LangChain pipeline", "B) Validate Model Connections", "C) Re-upload Document", "D) View Terminal Output"], 
                "correct": 0, 
                "explanation": f"System error layout: {str(e)}"
            }
        ]
    
def generate_learning_path(performance: dict, weaknesses: list, forgetting: list, student_name: str) -> dict:
    perf_sum = {s: {t: round(sum(a["pct"] for a in att) / len(att), 1) for t, att in tops.items()} for s, tops in performance.items()}
    
    template = """You are an AI tutor for {student_name}. Performance: {perf_json}. Weak: {weak_json}. Forgetting: {forget_json}.
Create a 7-day study plan. Return ONLY valid JSON matching this schema:
{{"path":[{{"day":1,"focus":"Topic","subject":"Subject","activity":"Action","duration_min":30,"priority":"high"}}],"insight":"One encouraging insight"}}"""
    
    prompt = PromptTemplate(input_variables=["student_name", "perf_json", "weak_json", "forget_json"], template=template)
    chain = prompt | llm | JsonOutputParser()
    
    try:
        return chain.invoke({
            "student_name": student_name,
            "perf_json": json.dumps(perf_sum),
            "weak_json": json.dumps([w['label'] for w in weaknesses[:3]]),
            "forget_json": json.dumps([f['topic'] for f in forgetting[:3]])
        })
    except:
        return {"path": [{"day": 1, "focus": "Review", "subject": "General", "activity": "Practice", "duration_min": 30, "priority": "high"}], "insight": "Keep studying!"}

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name, email, style = data.get("name", "").strip(), data.get("email", "").strip(), data.get("style", "conceptual")
    if not email or not name: return jsonify({"ok": False, "error": "Name and Email are required"}), 400
    db = load_db()
    if email in db["users"]: return jsonify({"ok": False, "error": "Email already registered."})
    student_id = make_student_id(email)
    db["users"][email] = {"name": name, "style": style, "id": student_id}
    save_db(db)
    session["student_name"], session["student_email"], session["student_id"], session["learn_style"] = name, email, student_id, style
    return jsonify({"ok": True, "student_id": student_id, "name": name})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip()
    if not email: return jsonify({"ok": False, "error": "Email is required"}), 400
    db = load_db()
    if email not in db["users"]: return jsonify({"ok": False, "error": "Account not found."})
    user_data = db["users"][email]
    session["student_name"], session["student_email"], session["student_id"], session["learn_style"] = user_data["name"], email, user_data["id"], user_data["style"]
    return jsonify({"ok": True, "student_id": user_data["id"], "name": user_data["name"]})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success"})

# ─── Data Routes ──────────────────────────────────────────────────────────────
@app.route("/upload_notes", methods=["POST"])
def upload_notes():
    student_id = get_student_id()
    if 'file' not in request.files or not request.form.get('subject'):
        return jsonify({"status": "error", "error": "Missing file or subject name"}), 400
    
    file = request.files['file']
    subject = request.form.get('subject').strip()
    
    try:
        if file.filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text_content = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            text_content = file.read().decode('utf-8')
    except Exception as e:
        return jsonify({"status": "error", "error": f"Failed to read file: {str(e)}"}), 400

    if not text_content.strip():
        return jsonify({"status": "error", "error": "File is completely empty."}), 400

    # --- LANGCHAIN DOCUMENT PIPELINE (Chunking & Vector Store Creation) ---
    print("⏳ LangChain Pipeline: Splitting text and indexing into FAISS vector space...")
    
    # Initialize LangChain's smart splitting mechanism
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    text_chunks = text_splitter.split_text(text_content)
    
    # Wrap texts into LangChain Document configurations
    lc_documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Build a file-less, high-speed in-memory FAISS store instance via LangChain
    vector_store = FAISS.from_documents(lc_documents, embeddings)
    
    rag_key = f"{student_id}||{subject}"
    VECTOR_REGISTRY[rag_key] = vector_store  # Keep vector active inside our runtime registry

    db = load_db()
    if student_id not in db["notes"]: db["notes"][student_id] = {}
    db["notes"][student_id][subject] = text_content
    save_db(db)

    # Core Topic Generation with LangChain
    template = """Analyze the following text sample for the subject "{subject}".
Identify 4 to 6 sequential chapter topics/headings present in this summary.
Return ONLY a valid JSON list of strings representing the names.
Example: ["Introduction to Arrays", "Binary Trees", "Recursion"]
--- SAMPLE CONTEXT ---
{sample_text}"""

    prompt = PromptTemplate(input_variables=["subject", "sample_text"], template=template)
    chain = prompt | llm | JsonOutputParser()

    try:
        topics = chain.invoke({"subject": subject, "sample_text": text_content[:2000]})
        if not isinstance(topics, list): topics = ["Fundamentals", "Key Concepts"]
    except:
        topics = ["Fundamentals", "Key Concepts", "Advanced Review"]

    curriculum = get_user_curriculum(student_id)
    curriculum[subject] = {"icon": "📄", "color": "#7c6af7", "topics": topics}
    save_user_curriculum(student_id, curriculum)
    mb_store(f"Uploaded notes and vectorized via LangChain for {subject}", ["notes", subject, student_id])

    return jsonify({"status": "success", "topics": topics})

@app.route("/curriculum")
def curriculum_route():
    return jsonify(get_user_curriculum(get_student_id()))

@app.route("/quiz/<subject>/<topic>")
def quiz(subject, topic):
    student_id = get_student_id()
    style = session.get("learn_style", "conceptual")
    questions = generate_questions(subject, topic, "medium", style, student_id)
    return jsonify({"questions": questions, "difficulty": "medium", "style": style})

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    data = request.get_json()
    student_id = get_student_id()
    score = sum(1 for i, q in enumerate(data["questions"]) if i < len(data["answers"]) and data["answers"][i] == q["correct"])
    pct = round(score / len(data["questions"]) * 100, 1)
    record_quiz_result(student_id, data["subject"], data["topic"], score, len(data["questions"]))
    return jsonify({"score": score, "total": len(data["questions"]), "percentage": pct})

@app.route("/dashboard")
def dashboard():
    return jsonify(compute_analytics(get_student_performance(get_student_id())))

@app.route("/learning_path")
def learning_path():
    performance = get_student_performance(get_student_id())
    analytics = compute_analytics(performance)
    return jsonify(generate_learning_path(performance, analytics["weaknesses"], analytics["forgetting"], get_student_name()))

@app.route("/concept_map/<subject>")
def concept_map(subject):
    curriculum = get_user_curriculum(get_student_id())
    topics = curriculum.get(subject, {}).get("topics", ["Topic 1", "Topic 2"])
    nodes = [{"id": t.lower().replace(" ", "_"), "label": t, "level": (i % 3) + 1} for i, t in enumerate(topics)]
    edges = [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"], "label": "leads to"} for i in range(len(nodes) - 1)]
    return jsonify({"nodes": nodes, "edges": edges})

@app.route("/student_info")
def student_info():
    return jsonify({"name": get_student_name(), "student_id": get_student_id(), "style": session.get("learn_style", "conceptual")})

if __name__ == "__main__":
    app.run(debug=True, port=5000)