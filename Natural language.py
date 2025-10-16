from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer


Sentences=['We are using the Bag of Word model', 'Bag of Word model isused for extracting the features.']


vectorizer = CountVectorizer()
features_text = vectorizer.fit_transform(Sentences)
print(vectorizer.vocabulary_)




