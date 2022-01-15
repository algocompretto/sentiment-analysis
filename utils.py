import re

def preprocess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]', '', text)
    new_text = re.sub('rt', '', new_text)
    return new_text
