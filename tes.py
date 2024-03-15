import re
from collections import defaultdict
from math import log10, sqrt
import requests
from faker import Faker  # Install library faker dengan pip install faker

# Fungsi untuk melakukan pra-pemrosesan teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(["dan", "atau", "dalam", "pada", "dengan"])
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Fungsi untuk membuat dokumen secara acak menggunakan library faker
def generate_fake_documents(num_documents):
    fake = Faker()
    documents = [fake.text() for _ in range(num_documents)]
    return documents

# Fungsi untuk mendapatkan dokumen dari API (jika diperlukan)
def get_documents_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Jumlah dokumen yang akan di-generate atau diambil dari API
num_documents = 5000

# Ganti URL API dengan URL sebenarnya yang Anda gunakan
api_url = "https://example.com/api/documents"
api_documents = get_documents_from_api(api_url)

if api_documents:
    documents = api_documents
else:
    # Generate 5000 dokumen acak jika tidak menggunakan API
    documents = generate_fake_documents(num_documents)

# Membuat indeks dokumen dengan bobot TF
document_index = defaultdict(dict)

for idx, doc in enumerate(documents):
    tokens = preprocess_text(doc)
    term_freq = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1
    document_index[idx] = term_freq

# Menghitung IDF
total_number_of_documents = len(documents)
idf = defaultdict(float)

for doc_id, term_freq in document_index.items():
    for term in term_freq:
        idf[term] += 1

for term, doc_freq in idf.items():
    idf[term] = log10(total_number_of_documents / doc_freq)

# Menghitung TF-IDF
tfidf_index = defaultdict(dict)

for doc_id, term_freq in document_index.items():
    for term, tf in term_freq.items():
        tfidf_index[doc_id][term] = tf * idf[term]

# Kueri pencarian
query = "car"

# Pra-pemrosesan kueri
query_tokens = preprocess_text(query)

# Menghitung vektor TF-IDF untuk kueri
query_tfidf = defaultdict(float)

for term in query_tokens:
    if term in idf:
        query_tfidf[term] = query_tokens.count(term) * idf[term]

# Menghitung Cosine Similarity antara kueri dan dokumen
cosine_similarities = {}

for doc_id, doc_tfidf in tfidf_index.items():
    dot_product = sum(query_tfidf[term] * doc_tfidf[term] for term in query_tfidf if term in doc_tfidf)
    query_norm = sqrt(sum(query_tfidf[term] ** 2 for term in query_tfidf))
    doc_norm = sqrt(sum(doc_tfidf[term] ** 2 for term in doc_tfidf))

    # Memastikan panjang vektor tidak nol sebelum melakukan pembagian
    if query_norm != 0 and doc_norm != 0:
        cosine_similarity = dot_product / (query_norm * doc_norm)
    else:
        cosine_similarity = 0  # Mengatasi pembagian oleh nol

    cosine_similarities[doc_id] = cosine_similarity



# Membuat indeks dokumen dengan bobot TF
document_index = defaultdict(dict)

# Menampilkan hasil pencarian berdasarkan Cosine Similarity
sorted_results = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 Dokumen Berdasarkan Cosine Similarity:")
for doc_id, similarity in sorted_results:
    print(f"Dokumen {doc_id}: Similarity = {similarity:.4f}")

# Menampilkan top 3 dokumen
top_3_results = sorted_results[:3]
print("\nTop 3 Dokumen:")
for doc_id, similarity in top_3_results:
    print(f"Dokumen {doc_id}: Similarity = {similarity:.4f}")
    print(f"Isi Dokumen: {documents[doc_id]}\n")
# Write your code here :-)
