import streamlit as st
from fastai.collab import load_learner
import pandas as pd
import os
import gdown
import torch

# File IDs from Google Drive
BOOKS_FILE_ID = "1-0VnAMedgcpjttaYTMUwSYeVpPqPOzd8"
MODEL_FILE_ID = "1PP1hTegxb1XFk_OLdRjiV9CTocjU9RJR"

# File paths
BOOKS_PATH = "Books.csv"
MODEL_PATH = "book_recommender.pkl"

# Function to download a file from Google Drive (only runs if file is missing)
def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Download files only if they don't exist
download_file(BOOKS_FILE_ID, BOOKS_PATH)
download_file(MODEL_FILE_ID, MODEL_PATH)

# Cache the dataset to prevent reloading
@st.cache_resource
def load_data():
    books_df = pd.read_csv(BOOKS_PATH)
    # Preprocess data (e.g., drop missing titles)
    books_df = books_df.dropna(subset=['Book-Title'])
    return books_df

# Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_model():
    return load_learner(MODEL_PATH)

# Load resources
books_df = load_data()
learn = load_model()

# Extract unique book titles
book_titles = books_df['Book-Title'].dropna().unique().tolist()

# Function to get recommendations
def get_similar_books(book_name, n=5):
    try:
        # Get the index of the selected book
        book_idx = learn.dls.classes['Book-Title'].o2i[book_name]
        
        # Get the book embeddings
        book_factors = learn.model.i_weight.weight
        
        # Calculate cosine similarity
        distances = torch.nn.functional.cosine_similarity(
            book_factors, book_factors[book_idx][None], dim=1
        )
        
        # Get the indices of the most similar books
        similar_book_indices = distances.argsort(descending=True)[1:n+1]  # Exclude the book itself
        
        # Get the titles of the similar books
        similar_books = [learn.dls.classes['Book-Title'][i] for i in similar_book_indices]
        return similar_books
    except KeyError:
        return ["Book not found in the model."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Streamlit UI
st.title("📚 Book Recommendation System")

# Use a text input box with dynamic filtering
search_query = st.text_input("Search for a book:", placeholder="Type the name of a book...")

# Filter book titles based on the search query
filtered_books = [title for title in book_titles if search_query.lower() in title.lower()]

# Display filtered books in a dropdown
if filtered_books:
    selected_book = st.selectbox("Choose a book:", filtered_books, index=None)
else:
    selected_book = None
    st.warning("No books found. Try a different search.")

# Button to generate recommendations
if st.button("Get Recommendations"):
    if selected_book:
        with st.spinner("Finding similar books..."):
            st.subheader(f"📖 Recommended Books for: {selected_book}")
            recommendations = get_similar_books(selected_book)
            for book in recommendations:
                st.write(f"- {book}")
    else:
        st.warning("Please select a book!")