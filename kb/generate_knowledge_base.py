import os
import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Make sure NLTK stuff is downloaded
nltk.download('punkt')

# --- Function to extract top keywords ---
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = np.array(tfidf_matrix.toarray()).flatten()
    keywords = np.array(vectorizer.get_feature_names_out())[scores.argsort()[::-1]]
    return keywords[:top_n]

# --- Read all .txt files from a folder ---
def read_text_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            topic = os.path.splitext(filename)[0]
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                data.append((topic, content))
    return data

# --- Split content into subtopics and form Q&A ---
def generate_kb(folder_path):
    files = read_text_files(folder_path)
    kb = []

    for topic, content in files:
        subtopics = extract_keywords(content, top_n=10)
        sentences = sent_tokenize(content)

        for subtopic in subtopics:
            # Find relevant sentences (basic match)
            matched_sentences = [s for s in sentences if subtopic.lower() in s.lower()]

            if not matched_sentences:
                continue

            answer = " ".join(matched_sentences[:5])  # pick top 3 relevant lines
            question = f"What is {subtopic}?"  # auto-generated question

            kb.append({
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "answer": answer
            })

    return kb

# --- Save KB as JSON ---
def save_kb_to_json(kb, output_file="kb/knowledge_base.json"):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=4)
    print(f"[✔] Knowledge base saved to {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    folder_path = "dataset_gfg_dbms"  # change this to your actual folder path
    kb = generate_kb(folder_path)
    save_kb_to_json(kb)
