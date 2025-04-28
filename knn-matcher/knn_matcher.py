import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# --- Load KB ---
with open('C:\AI-Mentor\kb\knowledge_base1.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

# --- Load Sentence Transformer Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Prepare KB Embeddings ---
kb_embeddings = []
kb_data = []
for entry in kb:
    embedding = model.encode(entry['question'])  # you can also use subtopic here
    kb_embeddings.append(embedding)
    kb_data.append(entry)

# --- Function to extract keywords (basic version using token frequency) ---
def extract_keywords(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalnum() and w.lower() not in nltk.corpus.stopwords.words('english')]
    return list(set(words))

# --- Function to find best matching KB entry ---
def find_best_match(user_input):
    keywords = extract_keywords(user_input)
    if not keywords:
        return None, None

    match_scores = []
    for kw in keywords:
        kw_vec = model.encode(kw)
        sims = cosine_similarity([kw_vec], kb_embeddings)[0]
        best_idx = np.argmax(sims)
        match_scores.append((sims[best_idx], kb_data[best_idx]))

    best_match = max(match_scores, key=lambda x: x[0]) if match_scores else (0, None)
    return best_match[1], best_match[0]

# --- Conversation Loop ---
def chat():
    print("🤖 Hi! Ask me anything or tell me something about yourself.")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("🤖 Goodbye!")
            break

        match, score = find_best_match(user_input)

        if match and score > 0.5:
            follow_up = f"Interesting! {match['question']}"
            print(f"🤖 {follow_up}\n➡️ {match['answer']}")
        else:
            print("🤖 Hmm, I couldn't find a good match for that. Could you rephrase?")

# --- Run Chat ---
if __name__ == '__main__':
    chat()
