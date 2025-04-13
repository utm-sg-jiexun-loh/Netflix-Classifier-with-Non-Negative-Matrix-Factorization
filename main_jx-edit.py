#%% 
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition._nmf import _initialize_nmf  # For NNDSVD initialization

def custom_nmf(V, n_components, max_iter=1000, tol=1e-5, seed=None):
    """
    Custom NMF implementation with scikit-learn compatibility
    """
    def update_H(W, H, V):
        numerator = W.T @ V
        denominator = (W.T @ W @ H) + 1e-10
        return H * (numerator / denominator)

    def update_W(W, H, V):
        numerator = V @ H.T
        denominator = (W @ H @ H.T) + 1e-10
        return W * (numerator / denominator)

    def _normalize(W, H):
        """Match scikit-learn's L1 normalization"""
        norms = np.linalg.norm(W, axis=0)
        W_normalized = W / norms
        H_normalized = H * norms[:, np.newaxis]
        return W_normalized, H_normalized

    # Initialize with NNDSVD
    W, H = _initialize_nmf(V, n_components, init='nndsvd', random_state=seed)
    
    prev_error = float('inf')
    
    for i in range(max_iter):
        H = update_H(W, H, V)
        W = update_W(W, H, V)
        
        # Critical normalization step
        W, H = _normalize(W, H)
        
        # Reconstruction error calculation
        WH = W @ H
        # Compute Frobenius norm of the error
        current_error = np.linalg.norm(V - WH, 'fro')
        print(f"Iteration {i+1}, Error: {current_error}")
  
        # Check convergence
        if prev_error - current_error < tol * prev_error:
            break
        prev_error = current_error

    return W, H

if __name__ == "__main__":
    # Data loading and preprocessing
    if os.path.exists('./data/netflix_titles.csv'):
        npr = pd.read_csv("./data/netflix_titles.csv")
        print("Available columns:", npr.columns)
        
        # Use the same preprocessing as scikit-learn version
        tfidf = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        dtm = tfidf.fit_transform(npr['description'])
        
        # Apply custom NMF
        W, H = custom_nmf(dtm, n_components=5, max_iter=1000, tol=1e-4, seed=42)
        
        # Display topics with weights (matches scikit-learn's format)
        for topic_idx, topic in enumerate(H):
            print(f"\nTopic #{topic_idx}:")
            top_indices = topic.argsort()[-15:][::-1]
            top_words = tfidf.get_feature_names_out()[top_indices]
            top_weights = topic[top_indices]
            
            for word, weight in zip(top_words, top_weights):
                print(f"{word}: {weight:.4f}")

    else:
        print("netflix_titles.csv not found")
        exit(1)
