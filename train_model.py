import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download('stopwords')
df = pd.read_csv("customer_support_tickets.csv")

print("Columns in dataset:", df.columns)

text_col1 = "Ticket Subject"
text_col2 = "Ticket Description"
target_col = "Ticket Type"

df['combined'] = df[text_col1] + " " + df[text_col2]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df['cleaned'] = df['combined'].apply(clean_text)

print("\nClass distribution:\n", df[target_col].value_counts())


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

y = df[target_col]

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("cat_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model retrained and saved successfully")