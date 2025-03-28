# Optim_proj

This project is designed to process text data using TF-IDF and NMF (Non-negative Matrix Factorization) to extract topics from a dataset. The main goal is to identify and analyze the most significant words associated with each topic.

## Files in the Project

### `optim.py`
This script contains the main logic for processing text data. It performs the following tasks:
- Reads a CSV file containing text data (e.g., movie descriptions).
- Applies TF-IDF vectorization to convert the text data into a numerical format.
- Fits an NMF model to the TF-IDF matrix to extract topics.
- Prints the top words and their corresponding weights for each identified topic.

### `generate_topics_csv.py`
This script is intended to generate a CSV file that contains the top words and their corresponding weights for each topic. The output CSV will have column names formatted as "topic1 weights1", "topic2 weights2", ..., "topicK weightsK".

## Setup Instructions

1. Ensure you have Python installed on your machine.
2. Install the required libraries by running:
   ```
   pip install pandas scikit-learn
   ```
3. Place your CSV file (e.g., `netflix_titles.csv`) in the project directory or update the file path in `optim.py` accordingly.

## Usage

1. Run `optim.py` to process the text data and view the top words for each topic:
   ```
   python optim.py
   ```
2. Run `generate_topics_csv.py` to create a CSV file with the top words and weights for each topic:
   ```
   python generate_topics_csv.py
   ```

## License
This project is licensed under the MIT License.