import re
import nltk
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def scan_resume(resume_path, jd_path):
    # Read files
    resume_text = read_pdf(resume_path) if resume_path.endswith(".pdf") else read_txt(resume_path)
    jd_text = read_txt(jd_path)

    # Clean text
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean, jd_clean])

    # Similarity
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    match_percent = round(score * 100, 2)

    # Top matching keywords
    feature_names = vectorizer.get_feature_names_out()
    resume_vec = tfidf[0].toarray()[0]
    jd_vec = tfidf[1].toarray()[0]

    common = [(feature_names[i], min(resume_vec[i], jd_vec[i]))
              for i in range(len(feature_names)) if resume_vec[i] > 0 and jd_vec[i] > 0]
    common = sorted(common, key=lambda x: x[1], reverse=True)[:15]

    print("\n📊 Resume Match Score:", match_percent, "%")
    print("\n🔑 Top Matched Keywords:")
    for w, _ in common:
        print("-", w)

if __name__ == "__main__":
    resume_path = "resume_job.txt"   # or resume.txt
    jd_path = "job_description.txt"
    scan_resume(resume_path, jd_path)
