import os
import io
import base64

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def make_dendrogram(ingredient_texts, labels, method='ward', metric='euclidean', num_colors=6):
    """
    ingredient_texts : array-like of strings (ingredients per recipe)
    labels : list of recipe names (leaf labels)
    returns base64 PNG image string of the dendrogram
    """
    # Vectorize ingredients (TF-IDF reduces common token weight)
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    X = vect.fit_transform(ingredient_texts).toarray()

    # Compute linkage
    Z = linkage(X, method=method, metric=metric)

    # Choose color threshold to define clusters visually
    max_d = np.max(Z[:, 2])
    color_threshold = max_d * 0.6  # adjust 0.6 for more/less clustering

    # Plot dendrogram with colored branches
    plt.figure(figsize=(12, 6))
    dend = dendrogram(
        Z,
        labels=labels,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=color_threshold,
        above_threshold_color='#777777',
        distance_sort='descending',
    )

    plt.title("Food Recipe Flavor Clusters (Hierarchical)")
    plt.xlabel("Recipes")
    plt.ylabel("Distance")
    plt.tight_layout()

    # Encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # file validation
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.csv'):
        return render_template('index.html', error="Please upload a CSV file.")

    # Save uploaded file
    fname = file.filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(path)

    # Read CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return render_template('index.html', error=f"Failed to read CSV: {str(e)}")

    # Expect columns: at least 'ingredients'. Optional 'title' or 'name'
    if 'ingredients' not in df.columns:
        return render_template('index.html', error="CSV must contain an 'ingredients' column.")

    # Prepare labels: prefer 'title' or 'name', else use index
    if 'title' in df.columns:
        labels = df['title'].astype(str).tolist()
    elif 'name' in df.columns:
        labels = df['name'].astype(str).tolist()
    else:
        # create short labels from first words of ingredient text if no title/name
        labels = [f"R{idx}" for idx in df.index]

    ingredient_texts = df['ingredients'].astype(str).tolist()

    try:
        img_base64 = make_dendrogram(ingredient_texts, labels)
    except Exception as e:
        return render_template('index.html', error=f"Error during clustering: {str(e)}")

    return render_template('result.html', dendrogram_img=img_base64, file_name=fname)

if __name__ == '__main__':
    app.run(debug=True)
