import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import argparse
import os

# ==== Utilities ====
def ensure_text_column(df):
    # Prioritaskan 'full_text', fallback ke 'text_for_tfidf'
    if 'full_text' in df.columns:
        return 'full_text'
    elif 'text_for_tfidf' in df.columns:
        return 'text_for_tfidf'
    else:
        # Coba tebak kolom teks terpanjang
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
        if text_cols:
            # pilih yang paling banyak kata rata-rata
            best = max(text_cols, key=lambda c: df[c].astype(str).str.split().str.len().mean())
            return best
        raise ValueError("Kolom teks tidak ditemukan. Pastikan ada 'full_text' atau 'text_for_tfidf'.")

def clean_label_series(s):
    # Normalisasi label umum: 1/0, pos/neg, Positive/Negative
    mapping = {
        'positive': 1, 'negatif': 0, 'negative': 0, 'positif': 1, 'pos': 1, 'neg': 0,
        'Positif': 1, 'Negatif': 0, 'Positive': 1, 'Negative': 0,
        'POSITIVE': 1, 'NEGATIVE': 0
    }
    s_norm = s.copy()
    if s_norm.dtype == 'O':
        s_norm = s_norm.fillna("").astype(str).str.strip()
        s_norm = s_norm.replace(mapping)
    # Pastikan biner 0/1
    uniq = pd.Series(s_norm.unique())
    # Bila ada selain {0,1}, map >0 to 1 else 0
    if not set(pd.unique(s_norm)).issubset({0,1}):
        try:
            s_norm = pd.to_numeric(s_norm, errors='coerce').fillna(0).astype(int)
            s_norm = (s_norm>0).astype(int)
        except Exception:
            # Fallback: treat all else as negative if contains "neg"
            s_norm = s.fillna("").astype(str).str.lower().str.contains("neg").astype(int)
    return s_norm.astype(int)

def make_wordcloud(texts, max_words=200, filename="wordcloud.png"):
    text_concat = " ".join(map(str, texts))
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text_concat)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc)
    ax.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Wordcloud disimpan sebagai {filename}")
    return filename

def top_frequent_words(texts, top_n=20):
    # Campuran stopwords ID+EN sederhana
    stop_id = {
        "yang","dan","di","ke","dari","untuk","pada","dengan","para","akan","atau",
        "itu","ini","karena","agar","saja","sudah","belum","bukan","adalah","dalam",
        "kami","kita","mereka","dia","ia","saya","aku","kamu","Anda","nya","akan","tidak",
        "ya","nih","dong","lah","pun","sebagai","juga","jadi","kalau","kalo","gak","ga"
    }
    vectorizer = CountVectorizer(stop_words=list(stop_id)+["the","a","an","and","or","is","are","to","of","for","in","on","at","be","with","as","this","that"])
    X = vectorizer.fit_transform([str(t) for t in texts])
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(freqs)[::-1][:top_n]
    return pd.DataFrame({"word": vocab[order], "count": freqs[order]})

def plot_confusion(cm, title="Confusion Matrix", filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Negatif","Positif"], yticklabels=["Negatif","Positif"], xlabel="Prediksi", ylabel="Aktual", title=title)
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix disimpan sebagai {filename}")
    return filename

def export_html_report(context):
    # Bangun HTML sederhana dengan gambar base64
    html_parts = []
    html_parts.append(f"<h1>{context['title']}</h1>")
    html_parts.append(f"<p><b>Total data:</b> {context['n_rows']}</p>")
    html_parts.append(f"<h2>Distribusi Sentimen</h2><pre>{context['label_counts_html']}</pre>")
    # Figures (wordcloud + confusions) disimpan sebagai base64
    for name, fig_path in context['figures']:
        with open(fig_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("ascii")
        html_parts.append(f"<h3>{name}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;'/>")
    # Accuracies
    html_parts.append("<h2>Hasil Akurasi</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>SVM (LinearSVC) — Accuracy: {context['svm_acc']:.4f} | CV (5-fold) mean: {context['svm_cv_mean']:.4f} ± {context['svm_cv_std']:.4f}</li>")
    html_parts.append(f"<li>Naive Bayes (MultinomialNB) — Accuracy: {context['nb_acc']:.4f} | CV (5-fold) mean: {context['nb_cv_mean']:.4f} ± {context['nb_cv_std']:.4f}</li>")
    html_parts.append("</ul>")
    # Save
    html = "<html><head><meta charset='utf-8'><title>Report SVM vs NB</title></head><body style='font-family:sans-serif;'>" + "".join(html_parts) + "</body></html>"
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Berhasil membuat file report.html di folder kerja aplikasi.")
    return "report.html"

def plot_distribution(y, filename="sentiment_distribution.png"):
    counts = y.value_counts().rename(index={0:"Negatif",1:"Positif"})
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Distribusi sentimen disimpan sebagai {filename}")
    return filename, counts

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Analisis Sentimen Kanker Payudara - SVM vs Naive Bayes')
    parser.add_argument('--file', type=str, default="Hasil Preprocessing Data 1 Kanker Payudara.csv", 
                        help='Path ke file CSV dataset')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proporsi data testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random state untuk reproducibility (default: 42)')
    parser.add_argument('--cv_folds', type=int, default=5, 
                        help='Jumlah fold untuk cross-validation (default: 5)')
    parser.add_argument('--top_n', type=int, default=20, 
                        help='Jumlah kata teratas untuk ditampilkan (default: 20)')
    
    args = parser.parse_args()
    
    # Judul program
    print("="*80)
    print("Perbandingan Analisis Kesadaran Kanker Payudara via X (Twitter)")
    print("SVM vs Naive Bayes")
    print("="*80)
    
    # ==== Data input ====
    print(f"\nMembaca data dari: {args.file}")
    try:
        df = pd.read_csv(args.file)
        print(f"Data berhasil dibaca. Jumlah baris: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File '{args.file}' tidak ditemukan.")
        print("Pastikan file berada di direktori yang benar atau gunakan argumen --file untuk menentukan path.")
        return
    
    # Validasi kolom
    if 'Label' not in df.columns:
        print("Error: Kolom 'Label' tidak ditemukan. Mohon pastikan dataset memiliki kolom 'Label' (positif/negatif atau 1/0).")
        return

    text_col = ensure_text_column(df)
    print(f"Kolom teks yang digunakan: '{text_col}'")

    # Bersihkan label
    y = clean_label_series(df['Label'])
    X_text = df[text_col].astype(str)

    # ==== Wordcloud & frequent words ====
    print("\n1. Membuat Wordcloud...")
    wc_filename = make_wordcloud(X_text)
    
    print(f"\n2. {args.top_n} Kata yang Paling Sering Muncul:")
    top_df = top_frequent_words(X_text, top_n=args.top_n)
    print(top_df.to_string(index=False))

    # ==== Distribusi sentimen ====
    print("\n3. Distribusi Sentimen:")
    dist_filename, counts = plot_distribution(y)
    print(counts.to_string())

    # ==== Split data ====
    print(f"\n4. Membagi data (test_size={args.test_size}, random_state={args.random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    # ==== Pipelines ====
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    pipe_svm = Pipeline([("tfidf", tfidf), ("clf", LinearSVC())])
    pipe_nb  = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])

    # ==== Train & evaluate SVM ====
    print("\n5. Melatih model SVM...")
    pipe_svm.fit(X_train, y_train)
    y_pred_svm = pipe_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_svm_filename = plot_confusion(cm_svm, "Confusion Matrix — SVM", "confusion_svm.png")

    # ==== Train & evaluate NB ====
    print("\n6. Melatih model Naive Bayes...")
    pipe_nb.fit(X_train, y_train)
    y_pred_nb = pipe_nb.predict(X_test)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_nb_filename = plot_confusion(cm_nb, "Confusion Matrix — Naive Bayes", "confusion_nb.png")

    print(f"\n7. Hasil Akurasi:")
    print(f"   SVM (LinearSVC): {svm_acc:.4f}")
    print(f"   Naive Bayes (MultinomialNB): {nb_acc:.4f}")

    # ==== Cross-validation ====
    print(f"\n8. Cross-Validation ({args.cv_folds}-fold):")
    svm_cv_scores = cross_val_score(pipe_svm, X_text, y, cv=args.cv_folds, n_jobs=-1)
    nb_cv_scores = cross_val_score(pipe_nb, X_text, y, cv=args.cv_folds, n_jobs=-1)

    print(f"   SVM — mean: {svm_cv_scores.mean():.4f} ± {svm_cv_scores.std():.4f}")
    print(f"   Naive Bayes — mean: {nb_cv_scores.mean():.4f} ± {nb_cv_scores.std():.4f}")

    # ==== Export HTML report ====
    print("\n9. Membuat laporan HTML...")
    context = {
        "title": "Laporan SVM vs Naive Bayes — Kesadaran Kanker Payudara (X/Twitter)",
        "n_rows": len(df),
        "label_counts_html": counts.to_string(),
        "svm_acc": svm_acc,
        "nb_acc": nb_acc,
        "svm_cv_mean": svm_cv_scores.mean(), 
        "svm_cv_std": svm_cv_scores.std(),
        "nb_cv_mean": nb_cv_scores.mean(), 
        "nb_cv_std": nb_cv_scores.std(),
        "figures": [
            ("Wordcloud", wc_filename),
            ("Distribusi Sentimen", dist_filename),
            ("Confusion Matrix — SVM", cm_svm_filename),
            ("Confusion Matrix — Naive Bayes", cm_nb_filename)
        ]
    }
    html_report_path = export_html_report(context)
    
    print("\n" + "="*80)
    print("ANALISIS SELESAI!")
    print(f"Laporan lengkap disimpan sebagai: {html_report_path}")
    print("File yang dihasilkan:")
    print(f"  - {wc_filename} (Wordcloud)")
    print(f"  - {dist_filename} (Distribusi Sentimen)")
    print(f"  - {cm_svm_filename} (Confusion Matrix SVM)")
    print(f"  - {cm_nb_filename} (Confusion Matrix Naive Bayes)")
    print(f"  - {html_report_path} (Laporan HTML lengkap)")
    print("="*80)

if __name__ == "__main__":
    main()