import faiss
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')

import numpy as np
import pandas as pd
import json

from sqlalchemy import create_engine

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_data(table_name, engine):
    data = pd.read_sql(
        f"SELECT * FROM {table_name}",
        con=engine)
    return data


def connect_db():
    db_user = 'admin'
    db_password = 'admin123'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'symentic_analysis'
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine


def search_semantic(query, index, df_embed, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec), top_k)
    return df_embed.iloc[I[0]][['Complaint_No', 'product', 'narrative', 'sentiment_label', 'sentiment_score']]

def main(query):
    engine = connect_db()
    df_embed = get_data('complaints_embedded_data', engine)
    df_embed['embedding'] = df_embed['embedding_json'].apply(json.loads)
    embeddings = np.vstack(df_embed['embedding'].to_numpy())
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
    index.add(embeddings)
    engine.dispose()
    data = search_semantic(query, index, df_embed, top_k=5)
    return data
