# Rajesh Pal Spam Email Classifier

A personal Python and Flask web app that classifies whether a submitted email subject is likely to be **Spam** or **Not Spam**.

This project is built and maintained by **Rajesh Pal** using a scikit-learn text pipeline and a Flask UI.

---

## Project Goal

The purpose of this app is to demonstrate a complete spam-classification workflow:

- read an SMS/email message sample dataset
- normalize and clean text
- train a machine learning classifier
- expose the classifier through a small Flask web interface

---

## What This App Does

A user enters an email subject line into the form shown by the web page. The app sends the text through a trained TF-IDF + LinearSVC classifier pipeline and returns **Spam** or **Not Spam**.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend and model workflow |
| Flask | Web framework |
| HTML/CSS | Frontend interface |
| scikit-learn | TF-IDF vectorizer, train/test split, LinearSVC |
| pandas | Data loading and dataset preparation |

---

## Project Structure

```text
Spam-Email-Classifier/
├── app.py                 # Flask app and model pipeline
├── spam.csv               # Dataset used for training
├── templates/
│   └── index.html         # Web form and UI
├── requirements.txt       # Python dependencies
├── model.ipynb            # Notebook for model experimentation
├── Sample.png              # UI screenshot
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
cd C:\Project
git clone <your-repository-url>
cd Spam-Email-Classifier
```

### 2. Install dependencies

```bash
C:\Users\rajes\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

### 3. Run the app

```bash
C:\Users\rajes\AppData\Local\Programs\Python\Python313\python.exe app.py
```

Then open:

http://127.0.0.1:5000

---

## Personalization Notes

This repository has been customized for **Rajesh Pal**. To make it truly your own, update:

- the repository name
- the project title and HTML header
- the README content
- the owner/contact information
- the dataset and training source if you use a different corpus

---

## Contact

Built by **Rajesh Pal**

Email: `rajeshpala552@gmail.com`

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
