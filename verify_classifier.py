import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df = df[df['label'].isin(['ham', 'spam'])].copy()
df['text'] = df['text'].fillna('').astype(str).apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('vectorizer', TfidfVectorizer(lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ('model', LinearSVC(class_weight='balanced')),
])

pipe.fit(X_train, y_train)
examples = [
    'rajeshpala552@gmail.com',
    'Free offer for your prize',
    'Important meeting today',
    'Mail update from the team',
    'Free event registration today',
    'Urgent meeting details',
]

for e in examples:
    print(e, '=>', 'spam' if pipe.predict([clean_text(e)])[0] == 1 else 'ham')

email = 'rajeshpala552@gmail.com'
cleaned_email = clean_text(email)
print('cleaned_email=', cleaned_email)
print('prediction=', 'spam' if pipe.predict([cleaned_email])[0] == 1 else 'ham')

features = pipe.named_steps['vectorizer'].get_feature_names_out()
feature_index = pipe.named_steps['vectorizer'].transform([cleaned_email]).indices
feature_values = pipe.named_steps['vectorizer'].transform([cleaned_email]).toarray()[0]
selected = {features[i]: feature_values[i] for i in feature_index if features[i] in {'gmail', 'com', 'mail', 'email', 'rajeshpala552', 'rajesh'}}
print('selected_features=', selected)

print('gmail_total_rows=', len(df[df['text'].str.contains('gmail', na=False)]))
print('gmail_spam_rows=', len(df[(df['label'] == 1) & df['text'].str.contains('gmail', na=False)]))
print('gmail_ham_rows=', len(df[(df['label'] == 0) & df['text'].str.contains('gmail', na=False)]))

print('class_counts:', y.value_counts().to_dict())
print('test_rows:', len(y_test))
