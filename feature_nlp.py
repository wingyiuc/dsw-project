import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from textblob import TextBlob


def preprocess_and_lemmatize(text):
    # Remove non-alphabetic characters and lowercase the text
    text = re.sub(r'\b\d+\b', '', text)  # Remove all standalone digits
    words = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    # Lemmatize each word
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(lemmatized_words)


# Function to extract keywords using TF-IDF
def extract_keywords(data, column, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(data[column])
    feature_names = tfidf.get_feature_names_out()
    dense = tfidf_matrix.todense()
    keywords = []
    for row in dense:
        word_scores = {feature_names[col]: row.item(col) for col in row.nonzero()[1]}
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        keywords.append(", ".join(word for word, score in sorted_words[:top_n]))
    return feature_names, keywords

def nlp(airbnb_data):
    """
    Processes Airbnb listing data to enhance with NLP features.

    Input:
    - airbnb_data: DataFrame with columns 'name' and 'description'.
    
    Output:
    - DataFrame with original data minus text columns, plus:
      - 'keywords_combined': Top 5 keywords from listings' text.
      - 'description_polarity': Sentiment polarity (-1 to 1).
      - 'description_subjectivity': Sentiment subjectivity (0 to 1).
    """
    
    # Check if required columns are present
    if 'name' not in airbnb_data.columns or 'description' not in airbnb_data.columns:
        return airbnb_data  # Return original DataFrame if required columns are missing

    # Step 1: Preprocess
    airbnb_data['combined_description'] = airbnb_data['name'].fillna('') + " " + airbnb_data['description'].fillna('')
    # Lemmatization takes a long time, so it's commented out
    # airbnb_data['combined_description'] = airbnb_data['combined_description'].apply(preprocess_and_lemmatize)

    # Step 2: Extract keywords
    feature_names, airbnb_data['keywords_combined'] = extract_keywords(airbnb_data, 'combined_description')
    # optional: Save feature names to CSV
    feature_names_df = pd.DataFrame(feature_names, columns=['Feature Name'])
    feature_names_df.to_csv('feature_names.csv', index=False)
    
    # Step 3: Perform sentiment analysis
    def get_sentiment(text):
        sentiment = TextBlob(text).sentiment
        return sentiment.polarity, sentiment.subjectivity

    airbnb_data[['description_polarity', 'description_subjectivity']] = airbnb_data['combined_description'].apply(get_sentiment).apply(pd.Series)

    # Drop unnecessary columns
    airbnb_data = airbnb_data.drop(columns=['name', 'description', 'combined_description'])

    return airbnb_data


"""
How to run the code: 

file_path = 'Airbnb_Data.csv'
df = pd.read_csv(file_path)
processed = nlp(df)
print(processed.head(3))
"""