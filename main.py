from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

from insert import insert_data
import transform
import get_sementic

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Optional, if used globally

EMBEDDED_PATH = 'data/embedded.csv'

# Load data into DB
def load_data():
    insert_data('complaints.csv', 'complaints')

# Run embedding + save to CSV
def embed_data():
    transform.main()

# Page 1: Admin page to load & embed
@app.route('/', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        if 'load' in request.form:
            print('Inserting data..')
            load_data()
        elif 'embed' in request.form:
            print('Embedding data..')
            embed_data()
        return redirect(url_for('admin'))

    return render_template('admin.html')

# Page 2: Semantic search
@app.route('/search', methods=['GET', 'POST'])
def search():
    query = ""
    df = pd.DataFrame()

    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            df = get_sementic.main(query)

    return render_template('search.html', query=query, table=df.to_html(classes='table table-hover', index=False) if not df.empty else None)

if __name__ == '__main__':
    app.run(debug=True)
