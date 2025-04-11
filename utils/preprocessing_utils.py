#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from unidecode import unidecode  # For stripping accents
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


##### uncomment the following lines if you need to install the libraries
# # Ensure NLTK resources are downloaded
# nltk.download('wordnet')
# nltk.download('stopwords')
##### uncomment the above lines if you need to install the libraries
# Helper function to map NLTK POS tags to WordNet POS tags
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer
    text = unidecode(text)  # Strip accents
    words = text.lower()
    pos_tags = pos_tag(word_tokenize(words))
    lemmatized_words = []
    for word, tag in pos_tags:
        if word not in stop_words:  # Remove stopwords
            # NN means noun, VB means verb, JJ means adjective
            if tag.startswith('NN'):
                lemmatized_words.append(lemmatizer.lemmatize(word, wordnet.NOUN))
            elif tag.startswith('VB'):
                lemmatized_words.append(lemmatizer.lemmatize(word, wordnet.VERB))
            elif tag.startswith('JJ'):
                lemmatized_words.append(lemmatizer.lemmatize(word, wordnet.ADJ))
            else:
                lemmatized_words.append(lemmatizer.lemmatize(word))  # Default to NOUN
    
    return ' '.join(lemmatized_words)  # Join words back into a single string

def preprocess_netflix_csv(csv_file_path="./data/netflix_titles.csv",save=True):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Preprocess the text in the 'description' column
    df['description'] = df['description'].fillna("").apply(preprocess_text)

    if save:
        # Save the processed DataFrame to a new CSV file
        df.to_csv("./data/processed_descriptions.csv", index=False)

        print("Processed descriptions saved to processed_descriptions.csv")

    else:
        print("Processed descriptions (first 5 rows):")
        print(df['description'].head())

        
    return df['description']  # Return the processed descriptions for further use

def vectorize_text(processed_descriptions,savestring="./data/filtered_one_hot_encoded.csv", min_occurrences=2):
    # Vectorize the processed descriptions into one-hot encoding
    vectorizer = CountVectorizer(binary=True)
    one_hot_matrix = vectorizer.fit_transform(processed_descriptions)

    # Convert to DataFrame for inspection
    one_hot_df = pd.DataFrame(one_hot_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Remove words with less than 2 occurrences
    word_counts = one_hot_df.sum(axis=0)  # Sum occurrences of each word across all rows
    filtered_words = word_counts[word_counts >= min_occurrences].index  # Keep words with 2 or more occurrences
    filtered_one_hot_df = one_hot_df[filtered_words]  # Filter the DataFrame to keep only these words

    # Save the filtered DataFrame to a CSV file
    filtered_one_hot_df.to_csv("./data/filtered_one_hot_encoded.csv", index=False)
    print(f"One-hot encoded DataFrame saved to {savestring}")

def load_one_hot_encoded_csv(csv_file_path="./data/filtered_one_hot_encoded.csv"):
    # Load the one-hot encoded CSV file
    df = pd.read_csv(csv_file_path)
    return df
# # Preprocessing: lemmatization, lowercasing, stopword removal, and accent stripping
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# print(f"to test if your function is running correctly: \nOriginal text: The brawler was running quickly.\n{preprocess_text("The brawler was running quickly.")}")
# print(preprocess_text("The brawler was running quickly. He ran pretty damned fast. He was faster than the wind"))  # Test the preprocessing function
# # Load the data

