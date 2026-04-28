"""
Resume Analyzer API - Production Level
Uses spaCy NLP + TF-IDF semantic similarity for intelligent skill matching
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import io
import re
import logging
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Load spaCy Model ─────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis and job match scoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Skills Taxonomy (Weighted by Category) ───────────────────────────────────
SKILL_TAXONOMY = {
    # Programming Languages - High importance
    "python": {"category": "Programming", "weight": 3, "aliases": ["py", "python3"]},
    "javascript": {"category": "Programming", "weight": 3, "aliases": ["js", "node", "nodejs", "node.js"]},
    "typescript": {"category": "Programming", "weight": 3, "aliases": ["ts"]},
    "java": {"category": "Programming", "weight": 3, "aliases": []},
    "c++": {"category": "Programming", "weight": 3, "aliases": ["cpp", "c plus plus"]},
    "c#": {"category": "Programming", "weight": 3, "aliases": ["csharp", "dotnet", ".net"]},
    "go": {"category": "Programming", "weight": 3, "aliases": ["golang"]},
    "rust": {"category": "Programming", "weight": 3, "aliases": []},
    "kotlin": {"category": "Programming", "weight": 2, "aliases": []},
    "swift": {"category": "Programming", "weight": 2, "aliases": []},
    "r": {"category": "Programming", "weight": 2, "aliases": ["r programming"]},
    "php": {"category": "Programming", "weight": 2, "aliases": []},
    "ruby": {"category": "Programming", "weight": 2, "aliases": []},
    "scala": {"category": "Programming", "weight": 2, "aliases": []},

    # Web Frameworks
    "react": {"category": "Frontend", "weight": 3, "aliases": ["reactjs", "react.js"]},
    "vue": {"category": "Frontend", "weight": 2, "aliases": ["vuejs", "vue.js"]},
    "angular": {"category": "Frontend", "weight": 2, "aliases": ["angularjs"]},
    "nextjs": {"category": "Frontend", "weight": 2, "aliases": ["next.js"]},
    "html": {"category": "Frontend", "weight": 2, "aliases": ["html5"]},
    "css": {"category": "Frontend", "weight": 2, "aliases": ["css3", "scss", "sass"]},
    "tailwind": {"category": "Frontend", "weight": 2, "aliases": ["tailwindcss", "tailwind css"]},

    # Backend Frameworks
    "fastapi": {"category": "Backend", "weight": 3, "aliases": ["fast api"]},
    "django": {"category": "Backend", "weight": 3, "aliases": []},
    "flask": {"category": "Backend", "weight": 2, "aliases": []},
    "spring": {"category": "Backend", "weight": 2, "aliases": ["spring boot", "springboot"]},
    "express": {"category": "Backend", "weight": 2, "aliases": ["expressjs", "express.js"]},
    "graphql": {"category": "Backend", "weight": 2, "aliases": []},
    "rest api": {"category": "Backend", "weight": 2, "aliases": ["restful", "rest", "restapi"]},

    # Data & AI
    "machine learning": {"category": "AI/ML", "weight": 3, "aliases": ["ml", "deep learning"]},
    "data science": {"category": "AI/ML", "weight": 3, "aliases": ["data scientist"]},
    "tensorflow": {"category": "AI/ML", "weight": 3, "aliases": ["tf"]},
    "pytorch": {"category": "AI/ML", "weight": 3, "aliases": ["torch"]},
    "scikit-learn": {"category": "AI/ML", "weight": 2, "aliases": ["sklearn"]},
    "pandas": {"category": "Data", "weight": 2, "aliases": []},
    "numpy": {"category": "Data", "weight": 2, "aliases": []},
    "data analysis": {"category": "Data", "weight": 2, "aliases": ["data analytics"]},
    "data visualization": {"category": "Data", "weight": 2, "aliases": ["tableau", "power bi"]},
    "statistics": {"category": "Data", "weight": 2, "aliases": ["statistical analysis"]},

    # Databases
    "sql": {"category": "Database", "weight": 3, "aliases": ["mysql", "postgresql", "postgres"]},
    "mongodb": {"category": "Database", "weight": 2, "aliases": ["mongo"]},
    "redis": {"category": "Database", "weight": 2, "aliases": []},
    "postgresql": {"category": "Database", "weight": 2, "aliases": ["postgres"]},
    "elasticsearch": {"category": "Database", "weight": 2, "aliases": []},

    # DevOps & Cloud
    "docker": {"category": "DevOps", "weight": 3, "aliases": ["containerization"]},
    "kubernetes": {"category": "DevOps", "weight": 3, "aliases": ["k8s"]},
    "aws": {"category": "Cloud", "weight": 3, "aliases": ["amazon web services"]},
    "gcp": {"category": "Cloud", "weight": 2, "aliases": ["google cloud", "google cloud platform"]},
    "azure": {"category": "Cloud", "weight": 2, "aliases": ["microsoft azure"]},
    "ci/cd": {"category": "DevOps", "weight": 2, "aliases": ["cicd", "github actions", "jenkins"]},
    "git": {"category": "DevOps", "weight": 2, "aliases": ["github", "gitlab", "version control"]},
    "linux": {"category": "DevOps", "weight": 2, "aliases": ["unix", "bash", "shell scripting"]},

    # Soft Skills
    "communication": {"category": "Soft Skills", "weight": 1, "aliases": []},
    "leadership": {"category": "Soft Skills", "weight": 1, "aliases": []},
    "agile": {"category": "Soft Skills", "weight": 2, "aliases": ["scrum", "kanban"]},
    "problem solving": {"category": "Soft Skills", "weight": 1, "aliases": []},
}

# Build alias lookup
ALIAS_LOOKUP = {}
for skill, info in SKILL_TAXONOMY.items():
    ALIAS_LOOKUP[skill] = skill
    for alias in info["aliases"]:
        ALIAS_LOOKUP[alias] = skill


# ─── Suggestion Engine ────────────────────────────────────────────────────────
SUGGESTION_TEMPLATES = {
    "Programming": {
        "python": "Build 2-3 real-world Python projects (API, data pipeline, automation). Contribute to open-source.",
        "javascript": "Master ES6+, async/await, and build a full-stack project using JavaScript across client and server.",
        "typescript": "Add TypeScript to your existing JS projects. Focus on interfaces, generics, and strict typing.",
        "java": "Work through core Java concepts, Spring Boot basics, and build a REST API with database integration.",
        "go": "Learn Go's concurrency model and build a microservice. Great for backend performance roles.",
    },
    "AI/ML": {
        "machine learning": "Complete Andrew Ng's ML course, implement algorithms from scratch, then apply to a Kaggle dataset.",
        "tensorflow": "Build end-to-end TF projects: data loading → model training → serving via API.",
        "pytorch": "Implement neural nets in PyTorch. Focus on autograd, custom datasets, and model deployment.",
        "data science": "Work on end-to-end EDA projects: data cleaning, feature engineering, modeling, and storytelling.",
    },
    "Frontend": {
        "react": "Build a multi-page React app with hooks, context API, and React Router. Deploy on Vercel.",
        "html": "Master semantic HTML5, accessibility (ARIA), and form handling best practices.",
        "css": "Learn CSS Grid, Flexbox, animations, and responsive design. Rebuild popular UI components.",
        "typescript": "Migrate a JS project to TypeScript. Focus on strict types, interfaces, and generics.",
    },
    "Backend": {
        "fastapi": "Build production-grade REST APIs with FastAPI: authentication, async routes, and OpenAPI docs.",
        "django": "Create a full Django project with authentication, REST API via DRF, and PostgreSQL.",
        "docker": "Containerize your existing apps, write multi-service docker-compose files.",
    },
    "Database": {
        "sql": "Practice complex SQL: window functions, CTEs, query optimization, and indexing strategies.",
        "mongodb": "Design document schemas, practice aggregation pipelines, and index optimization.",
        "postgresql": "Set up Postgres locally, practice constraints, transactions, and performance tuning.",
    },
    "DevOps": {
        "docker": "Containerize an existing project. Build a docker-compose stack with DB and cache.",
        "kubernetes": "Deploy a multi-service app to a local Minikube cluster. Learn pods, services, and ingress.",
        "aws": "Get AWS Cloud Practitioner certified. Build a serverless API with Lambda + API Gateway.",
        "git": "Practice Git workflows: branching strategies, rebasing, squashing, PR reviews, and CI integration.",
    },
}

def get_dynamic_suggestion(skill: str, category: str) -> str:
    """Generate suggestion, falling back to a generic template."""
    cat_suggestions = SUGGESTION_TEMPLATES.get(category, {})
    if skill in cat_suggestions:
        return cat_suggestions[skill]
    # Generic by category
    generic = {
        "Programming": f"Build 2-3 projects using {skill.title()}. Publish on GitHub with documentation.",
        "AI/ML": f"Take a structured course on {skill.title()} and apply it on a public dataset.",
        "Frontend": f"Build UI components using {skill.title()} and deploy a demo project.",
        "Backend": f"Create a REST API or microservice using {skill.title()}.",
        "Database": f"Design schemas and run complex queries with {skill.title()} on a sample dataset.",
        "DevOps": f"Set up a {skill.title()} pipeline for an existing project and document the workflow.",
        "Cloud": f"Earn a {skill.upper()} certification and deploy a real project to the cloud.",
        "Data": f"Complete an end-to-end analysis project using {skill.title()}.",
        "Soft Skills": f"Actively practice {skill.title()} in team projects, code reviews, and presentations.",
    }
    return generic.get(category, f"Study {skill.title()} through projects, tutorials, and documentation.")


# ─── Text Extraction ──────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes with error handling."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        full_text = " ".join(text_parts)
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or scanned (non-extractable)")
        return full_text
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text from PDF: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return " ".join([para.text for para in doc.paragraphs])
    except ImportError:
        raise HTTPException(status_code=422, detail="DOCX support requires python-docx")


# ─── NLP Skill Extractor ──────────────────────────────────────────────────────
class SkillExtractor:
    """
    Multi-strategy skill extractor:
    1. Alias keyword matching (with word boundaries)
    2. spaCy NER for custom entity detection
    3. TF-IDF semantic similarity for fuzzy matches
    """

    def __init__(self):
        self.skills = list(SKILL_TAXONOMY.keys())
        self.all_terms = list(ALIAS_LOOKUP.keys())

    def _keyword_match(self, text: str) -> set:
        """Match skills using word-boundary-aware regex."""
        found = set()
        text_lower = text.lower()
        for term, canonical in ALIAS_LOOKUP.items():
            # Use word boundaries; escape for special regex chars
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found.add(canonical)
        return found

    def _spacy_extract(self, text: str) -> set:
        """Use spaCy to extract named entities that may be skills."""
        if not nlp:
            return set()
        found = set()
        doc = nlp(text[:100000])  # spaCy limit safety
        for ent in doc.ents:
            ent_lower = ent.text.lower().strip()
            if ent_lower in ALIAS_LOOKUP:
                found.add(ALIAS_LOOKUP[ent_lower])
        return found

    def _semantic_match(self, text: str, threshold: float = 0.75) -> set:
        """Use TF-IDF cosine similarity for fuzzy/semantic skill detection."""
        found = set()
        try:
            # Split text into sentences/chunks for comparison
            sentences = re.split(r'[.\n,;]+', text.lower())
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            if not sentences:
                return found

            vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
            corpus = self.skills + sentences
            tfidf_matrix = vectorizer.fit_transform(corpus)

            skill_vecs = tfidf_matrix[:len(self.skills)]
            sent_vecs = tfidf_matrix[len(self.skills):]

            sims = cosine_similarity(skill_vecs, sent_vecs)  # [n_skills, n_sentences]
            for i, skill in enumerate(self.skills):
                if sims[i].max() >= threshold:
                    found.add(skill)
        except Exception as e:
            logger.warning(f"Semantic matching error: {e}")
        return found

    def extract(self, text: str) -> set:
        """Combine all strategies for best coverage."""
        keyword_results = self._keyword_match(text)
        spacy_results = self._spacy_extract(text)
        semantic_results = self._semantic_match(text)
        all_found = keyword_results | spacy_results | semantic_results
        return all_found


# ─── Weighted Scoring Engine ──────────────────────────────────────────────────
class ScoringEngine:
    """
    Calculates weighted match score based on skill importance levels.
    High-weight skills contribute more to the score.
    """

    def score(
        self,
        resume_skills: set,
        jd_skills: set
    ) -> dict:
        if not jd_skills:
            return {
                "match_score": 0,
                "weighted_score": 0,
                "matched_skills": [],
                "missing_skills": [],
                "score_breakdown": {}
            }

        matched = jd_skills & resume_skills
        missing = jd_skills - resume_skills

        # Weighted score
        total_weight = sum(SKILL_TAXONOMY.get(s, {}).get("weight", 1) for s in jd_skills)
        matched_weight = sum(SKILL_TAXONOMY.get(s, {}).get("weight", 1) for s in matched)

        weighted_score = int((matched_weight / total_weight) * 100) if total_weight > 0 else 0
        raw_score = int((len(matched) / len(jd_skills)) * 100)

        # Category breakdown
        breakdown = {}
        for skill in jd_skills:
            cat = SKILL_TAXONOMY.get(skill, {}).get("category", "Other")
            if cat not in breakdown:
                breakdown[cat] = {"required": 0, "matched": 0, "skills": []}
            breakdown[cat]["required"] += 1
            breakdown[cat]["skills"].append({
                "skill": skill,
                "matched": skill in matched,
                "weight": SKILL_TAXONOMY.get(skill, {}).get("weight", 1)
            })
            if skill in matched:
                breakdown[cat]["matched"] += 1

        # Add percentage per category
        for cat in breakdown:
            r = breakdown[cat]["required"]
            m = breakdown[cat]["matched"]
            breakdown[cat]["percentage"] = int((m / r) * 100) if r > 0 else 0

        return {
            "match_score": raw_score,
            "weighted_score": weighted_score,
            "matched_skills": sorted(list(matched)),
            "missing_skills": sorted(list(missing)),
            "score_breakdown": breakdown,
        }


# ─── Resume Profile Scorer ────────────────────────────────────────────────────
def score_resume_profile(resume_text: str, resume_skills: set) -> dict:
    """Score overall resume quality based on content signals."""
    text_lower = resume_text.lower()
    signals = {
        "has_quantified_achievements": bool(re.search(r'\d+%|\d+ years?|\$\d+|\d+[kK]\+?', resume_text)),
        "has_github_linkedin": bool(re.search(r'github\.com|linkedin\.com', text_lower)),
        "has_education": bool(re.search(r'bachelor|master|phd|degree|university|college|b\.tech|b\.e\.|m\.tech', text_lower)),
        "has_experience_section": bool(re.search(r'experience|work history|employment', text_lower)),
        "has_projects": bool(re.search(r'project|portfolio|built|developed|created', text_lower)),
        "skill_breadth": min(len(resume_skills), 15),
    }

    score = 0
    if signals["has_quantified_achievements"]: score += 25
    if signals["has_github_linkedin"]: score += 15
    if signals["has_education"]: score += 15
    if signals["has_experience_section"]: score += 20
    if signals["has_projects"]: score += 15
    score += int((signals["skill_breadth"] / 15) * 10)

    return {"resume_profile_score": min(score, 100), "signals": signals}


# ─── Singleton Instances ──────────────────────────────────────────────────────
extractor = SkillExtractor()
scorer = ScoringEngine()


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "running",
        "version": "2.0.0",
        "nlp_enabled": nlp is not None,
        "skills_in_taxonomy": len(SKILL_TAXONOMY)
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    include_semantic: Optional[bool] = Form(True)
):
    """
    Analyze a resume PDF/DOCX against a job description.

    Returns:
    - Weighted match score
    - Matched & missing skills with categories
    - Score breakdown by skill category
    - Dynamic suggestions with priority levels
    - Resume profile quality score
    """
    # ── Validate inputs ────────────────────────────────────────────────────────
    if not job_description.strip() or len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short (min 20 chars)")

    content_type = file.content_type or ""
    filename = file.filename or ""

    if not (
        "pdf" in content_type or
        "word" in content_type or
        filename.endswith(".pdf") or
        filename.endswith(".docx")
    ):
        raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported")

    # ── Extract text ───────────────────────────────────────────────────────────
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    try:
        if filename.endswith(".docx") or "word" in content_type:
            resume_text = extract_text_from_docx(file_bytes)
        else:
            resume_text = extract_text_from_pdf(file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    logger.info(f"Extracted {len(resume_text)} characters from resume")

    # ── Extract skills ─────────────────────────────────────────────────────────
    resume_skills = extractor.extract(resume_text)
    jd_skills = extractor.extract(job_description)

    logger.info(f"Resume skills: {len(resume_skills)}, JD skills: {len(jd_skills)}")

    if not jd_skills:
        raise HTTPException(
            status_code=422,
            detail="Could not detect any recognizable skills in the job description. Please provide a more detailed JD."
        )

    # ── Score ──────────────────────────────────────────────────────────────────
    result = scorer.score(resume_skills, jd_skills)
    profile = score_resume_profile(resume_text, resume_skills)

    # ── Generate dynamic suggestions ───────────────────────────────────────────
    suggestions = []
    missing_skills = result["missing_skills"]

    for skill in missing_skills:
        meta = SKILL_TAXONOMY.get(skill, {})
        category = meta.get("category", "General")
        weight = meta.get("weight", 1)

        if weight == 3:
            priority = "Critical"
        elif weight == 2:
            priority = "High"
        else:
            priority = "Medium"

        advice = get_dynamic_suggestion(skill, category)
        suggestions.append({
            "skill": skill,
            "category": category,
            "priority": priority,
            "weight": weight,
            "advice": advice,
        })

    # Sort: Critical first, then High, then Medium
    priority_order = {"Critical": 0, "High": 1, "Medium": 2}
    suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    # ── Build response ─────────────────────────────────────────────────────────
    return {
        "status": "success",
        "match_score": result["match_score"],
        "weighted_score": result["weighted_score"],
        "resume_profile_score": profile["resume_profile_score"],
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "resume_skills": sorted(list(resume_skills)),
        "job_required_skills": sorted(list(jd_skills)),
        "score_breakdown": result["score_breakdown"],
        "profile_signals": profile["signals"],
        "suggestions": suggestions,
        "meta": {
            "resume_length": len(resume_text),
            "nlp_enabled": nlp is not None,
            "skills_detected_in_resume": len(resume_skills),
            "skills_required_by_jd": len(jd_skills),
        }
    }


@app.get("/skills", tags=["Reference"])
def list_skills():
    """Return all skills in the taxonomy with metadata."""
    by_category = {}
    for skill, meta in SKILL_TAXONOMY.items():
        cat = meta["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "skill": skill,
            "weight": meta["weight"],
            "aliases": meta["aliases"]
        })
    return {"total": len(SKILL_TAXONOMY), "by_category": by_category}