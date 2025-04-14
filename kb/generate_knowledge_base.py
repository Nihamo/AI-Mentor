import os
import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK tokenizer model
nltk.download('punkt')

# File to keep track of processed files
PROCESSED_FILE_LOG = "kb/processed_files.txt"

# --- Function to extract top keywords ---
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = np.array(tfidf_matrix.toarray()).flatten()
    keywords = np.array(vectorizer.get_feature_names_out())[scores.argsort()[::-1]]
    return keywords[:top_n]

# --- Load already processed file names ---
def load_processed_files():
    if os.path.exists(PROCESSED_FILE_LOG):
        with open(PROCESSED_FILE_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

# --- Save newly processed file names ---
def save_processed_files(processed_files):
    os.makedirs(os.path.dirname(PROCESSED_FILE_LOG), exist_ok=True)
    with open(PROCESSED_FILE_LOG, 'a') as f:
        for file in processed_files:
            f.write(file + '\n')

# --- Read new .txt files from a folder ---
def read_text_files(folder_path, processed_files):
    data = []
    new_processed = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename not in processed_files:
            topic = os.path.splitext(filename)[0]
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin1') as f:
                    content = f.read()
            data.append((topic, content))
            new_processed.append(filename)
    return data, new_processed

# --- Generate knowledge base entries ---
def generate_kb(folder_path, processed_files):
    files, new_processed = read_text_files(folder_path, processed_files)
    kb = []

    for topic, content in files:
        subtopics = extract_keywords(content, top_n=10)
        sentences = sent_tokenize(content)

        for subtopic in subtopics:
            matched_sentences = [s for s in sentences if subtopic.lower() in s.lower()]
            if not matched_sentences:
                continue

            answer = " ".join(matched_sentences[:5])
            question = f"What is {subtopic}?"

            kb.append({
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "answer": answer
            })

    return kb, new_processed

# --- Save KB as JSON ---
def save_kb_to_json(kb, output_file="kb/knowledge_base1.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load existing KB if available
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_kb = json.load(f)
    else:
        existing_kb = []

    combined_kb = existing_kb + kb

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_kb, f, indent=4, ensure_ascii=False)
    print(f"[✔] {len(kb)} new entries added to {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    folder_path = "dataset_gfg_dbms"
    processed_files = load_processed_files()

    kb, newly_processed = generate_kb(folder_path, processed_files)

    if kb:
        save_kb_to_json(kb)
        save_processed_files(newly_processed)
    else:
        print("No new files to process.")
