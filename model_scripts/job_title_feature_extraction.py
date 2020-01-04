import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))
#stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten'])

def get_first_title(title):
    # keep "co-founder, co-ceo, etc"
    title = re.sub(r"[Cc]o[\-\ ]","", title)
    split_titles = re.split(r"\,|\-|\||\&|\:|\/|and|\(", title)
    return split_titles[0].strip()
def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
        stemSentence = stemSentence.strip()
    return stemSentence

def get_title_features(title):
    features = {}
    title = get_first_title(title)
    word_tokens = nltk.word_tokenize(title)
    filtered_words = [w for w in word_tokens if not w in stop_words]

    for word in filtered_words:
        features['contains({})'.format(word.lower())] = True
    if len(filtered_words) > 0:
        first_key = 'first({})'.format(filtered_words[0].lower())
        last_key = 'last({})'.format(filtered_words[-1].lower())
        features[first_key] = True
        features[last_key] = True
    return features