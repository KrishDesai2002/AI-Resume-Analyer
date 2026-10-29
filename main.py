"""AI Resume Analyzer - FastAPI backend.

Run:  pip install -r requirements.txt && uvicorn main:app --reload --port 8000
Then set VITE_API_URL=http://localhost:8000 in the frontend.
To plug in an AI model later, implement `ai_analyze` below.
"""
import io
import re
from collections import Counter

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Resume Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "nosql",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring",
    "html", "css", "tailwind", "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "pandas", "numpy",
    "scikit-learn", "data analysis", "power bi", "tableau", "excel", "mongodb", "postgresql",
    "mysql", "redis", "rest api", "graphql", "ci/cd", "agile", "scrum", "jira", "figma",
    "communication", "leadership", "teamwork", "problem solving", "testing", "microservices",
]
STOP = set("the and for with you are our will have from that this your about who what when into their they them able work team role job using use any all can more also must such other within across strong good great years year experience including etc per new well".split())


def extract_text(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if name.lower().endswith(".docx"):
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise HTTPException(400, "Only PDF and DOCX files are supported")


def has(text: str, term: str) -> bool:
    return re.search(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)", text, re.I) is not None


def keywords(text: str, n: int = 25):
    words = [w.rstrip(".") for w in re.findall(r"[a-z][a-z+#.]{2,}", text.lower())]
    return [w for w, _ in Counter(w for w in words if w not in STOP).most_common(n)]


def analyze(resume: str, jd: str) -> dict:
    r = resume.lower()
    res_skills = [s for s in SKILLS if has(r, s)]
    jd_skills = [s for s in SKILLS if has(jd, s)]
    matching = [s for s in jd_skills if s in res_skills]
    missing = [s for s in jd_skills if s not in res_skills]
    jd_kw, res_kw = keywords(jd), keywords(resume)
    missing_kw = [k for k in jd_kw if not has(r, k)]
    kw_hit = (len(jd_kw) - len(missing_kw)) / len(jd_kw) if jd_kw else 0
    skill_hit = len(matching) / len(jd_skills) if jd_skills else kw_hit
    match = round((skill_hit * 0.6 + kw_hit * 0.4) * 100)

    sections = ["experience", "education", "skills", "projects", "summary", "certifications"]
    found = [s for s in sections if has(r, s)]
    email = bool(re.search(r"\S+@\S+\.\S+", resume))
    phone = bool(re.search(r"\+?\d[\d\s-]{8,}", resume))
    nums = len(re.findall(r"\d+%|\$\d+|\d+\+", resume))
    words = len(resume.split())
    verbs = sum(v in r for v in ["developed", "built", "led", "designed", "implemented", "improved", "managed", "created", "optimized"])
    ats = 30 + len(found) * 8 + (8 if email else 0) + (6 if phone else 0) + (8 if 250 <= words <= 1000 else 0)
    ats = min(100, ats)
    overall = round(match * 0.5 + ats * 0.3 + min(100, nums * 15 + verbs * 8) * 0.2)

    strengths, weaknesses, suggestions = [], [], []
    if len(matching) >= 3: strengths.append(f"Relevant technical skills: {', '.join(matching[:4])}")
    if len(res_skills) >= 8: strengths.append("Broad and diverse skill set")
    if has(r, "projects"): strengths.append("Includes a dedicated projects section")
    if verbs >= 3: strengths.append("Uses strong action verbs to describe work")
    if nums >= 3: strengths.append("Contains measurable achievements")
    if missing: weaknesses.append(f"Missing job-required skills: {', '.join(missing[:5])}")
    if len(missing_kw) > 8: weaknesses.append("Many job description keywords are missing")
    if nums < 2: weaknesses.append("Few measurable achievements (numbers, %, impact)")
    if verbs < 2: weaknesses.append("Weak project/experience descriptions")
    if missing: suggestions.append(f"If you have experience with {', '.join(missing[:3])}, add it explicitly.")
    if missing_kw: suggestions.append(f"Weave these keywords in naturally: {', '.join(missing_kw[:6])}.")
    if nums < 3: suggestions.append("Quantify results, e.g. 'Reduced load time by 40%'.")
    suggestions.append("Use a single-column layout with standard fonts for ATS readability.")

    return {
        "overall_score": overall, "ats_score": ats, "match_score": match,
        "ats_feedback": "ATS-friendly" if ats >= 75 else "Moderately ATS-friendly" if ats >= 55 else "Likely to struggle with ATS",
        "matching_skills": matching, "missing_skills": missing, "resume_skills": res_skills,
        "strengths": strengths or ["Resume text is readable"], "weaknesses": weaknesses or ["No major weaknesses detected"],
        "suggestions": suggestions, "jd_keywords": jd_kw, "resume_keywords": res_kw, "missing_keywords": missing_kw,
    }


def ai_analyze(resume: str, jd: str, base: dict) -> dict:
    """Hook for a future AI API (e.g. call an LLM and merge its feedback into `base`)."""
    return base


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_endpoint(resume: UploadFile = File(...), job_description: str = Form(...)):
    if not job_description.strip():
        raise HTTPException(400, "Job description is required")
    text = extract_text(resume.filename or "", await resume.read())
    if len(text.strip()) < 50:
        raise HTTPException(422, "Could not extract text from the resume")
    return ai_analyze(text, job_description, analyze(text, job_description))
