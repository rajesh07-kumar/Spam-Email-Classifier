import re
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_pipeline():
    df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

    # Dataset column layout in this project is v1=label, v2=text
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'text']
    else:
        df = df[['label', 'text']].copy()

    # Keep only the two valid target classes
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    df['text'] = df['text'].fillna('').astype(str).apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ('model', LinearSVC(class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


pipeline = build_pipeline()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        cleaned_subject = clean_text(subject)
        prediction = pipeline.predict([cleaned_subject])
        session['result'] = "Spam" if prediction[0] == 1 else "Not Spam"
        return redirect(url_for('index'))

    result = session.pop('result', None)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
