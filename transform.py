import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
from transformers import pipeline

import pandas as pd
import re
import json

from sqlalchemy import create_engine

stop_words = set(stopwords.words('english'))
classifier = pipeline("sentiment-analysis")
model = SentenceTransformer('all-MiniLM-L6-v2')

def connect_db():
    db_user = 'admin'
    db_password = 'admin123'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'symentic_analysis'
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine

def get_data(table_name, engine):
    data = pd.read_sql(
    f"SELECT * FROM {table_name}",
    con=engine)
    return data

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)             # remove punctuation
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return ' '.join(filtered_words)

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', str(text))

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F" 
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF" 
        "\U00002500-\U00002BEF"  
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642" 
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"               
        "\u3030"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', str(text))

def get_semantic_score(text):
    if not text or pd.isnull(text):
        return {'label': 'NEUTRAL', 'score': 0.0}
    result = classifier(text[:512])[0]
    return result

def preprocess_narrative(text):
    text = remove_urls(text)
    text = remove_emojis(text)
    text = clean_text(text)
    return text

def main():
    engine = connect_db()
    data = get_data('complaints', engine)[:3000]
    data['cleaned_narrative'] = data['narrative'].fillna('').apply(preprocess_narrative)
    data['sentiment_result'] = data['cleaned_narrative'].apply(get_semantic_score)
    data['sentiment_label'] = data['sentiment_result'].apply(lambda x: x['label'])
    data['sentiment_score'] = data['sentiment_result'].apply(lambda x: x['score'])
    data['embedding'] = model.encode(data['cleaned_narrative'].fillna(''), normalize_embeddings=True).tolist()
    data['embedding_json'] = data['embedding'].apply(lambda x: json.dumps(x))

    column_list = ['Complaint_No', 'product', 'narrative', 'cleaned_narrative', 'sentiment_label', 'sentiment_score', 'embedding_json']
    data[column_list].to_sql('complaints_embedded_data', engine, if_exists='append', index=False)

    engine.dispose()