import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset.csv")

df = df[["url", "type"]].dropna()

df = df.sample(50000, random_state=42)

# ---------------- LABELS ----------------
label_map = {
    "benign": 0,
    "defacement": 1,
    "phishing": 2,
    "malware": 3
}

df["label"] = df["type"].map(label_map)

X = df["url"]
y = df["label"]

# ---------------- TRAIN ONLY ONCE ----------------
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- FUNCTION ----------------
def predict_url(url):
    url_vec = vectorizer.transform([url])

    pred = model.predict(url_vec)[0]
    prob = model.predict_proba(url_vec).max() * 100

    reverse_map = {
        0: "benign",
        1: "defacement",
        2: "phishing",
        3: "malware"
    }

    return reverse_map[pred], round(prob, 2)