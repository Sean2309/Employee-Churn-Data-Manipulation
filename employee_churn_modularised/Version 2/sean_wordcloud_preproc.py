# import nltk
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from nltk import ne_chunk


### START OF WORD CLOUD PRE PROCESSING

# def stem(text):
#     pst = PorterStemmer()
#     result = [pst.stem(w.lower()) for w in text.split()]
#     return ' '.join(result)

# def lemmatize(text):
#     lemmatizer = WordNetLemmatizer() 
#     result = [lemmatizer.lemmatize(w.lower()) for w in text.split()]
#     return ' '.join(result)

# def remove_stopwords_punctuations(text, stopwords, additional_sw=None):
#     # remove stopwords, punc, numerics
#     if additional_sw is not None: 
#         stopwords.append(additional_sw)
#     processed = [w.lower() for w in text.split(' ') if w.lower() not in stopwords and not w.isdigit() and w not in string.punctuation]
#     processed = ' '.join(processed)
#     return processed

# def preprocess_text(text, stopwords, additional_sw=None):
#     # text = stem(text)
#     text = remove_stopwords_punctuations(text, stopwords, additional_sw)
#     return text
    
### END OF WORD CLOUD PRE PROCESSING