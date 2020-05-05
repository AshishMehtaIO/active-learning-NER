# from google.colab import drive
# drive.mount('/content/drive')

# !pip install sklearn_crfsuite

from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier, RidgeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import string
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import sklearn_crfsuite
import pickle
import numpy as np
from sklearn.metrics import f1_score
from nltk.corpus.reader import ConllCorpusReader
import nltk
import matplotlib.pyplot as plt

nltk.download('conll2002')

regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def wordshape(text):

    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)

def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    features = [
        (str(o) + 'word', word)
        # TODO: add more features here.
    ]
    return features

def gettag(tag, o):
    features = [ (str(o) +"tag", tag) ]
    return features

def gethyphen(word, o):
    if('-' in word):
        features = [(str(o) +"hyphen", 1)]
    else:
        features = [(str(o) +"hyphen", 0)]
    return features
    
def capletter(word, o):
    if(word[0].isupper):
        features = [(str(o) +"first_upper", 1)]
    else:
        features = [(str(o) +"first_upper", 0)]
    return features

def noun_suffix(word, o):
    if(word.endswith('o') or word.endswith('or') or word.endswith('a') or word.endswith('ora')):
        features = [(str(o) +"common_suffix", 1)]
    else:
        features = [(str(o) +"common_suffix", 0)]
    return features

def get_wordshape(word, o):
    feature = [(str(o) +"word_shape", wordshape(word))]
    return feature

def all_upper(word, o):
    if(word.isupper()):
        return [(str(o) +"all_upper", 1)]
    else:
        return [(str(o) +"all_upper", 0)]

def all_lower(word, o):
    if(word.islower()):
        return [(str(o) +"all_lower", 1)]
    else:
        return [(str(o) +"all_lower", 0)]

def has_apostrophe(word, o):
    if("'" in word):
        return [(str(o) +"apostrophe", 1)]
    else:
        return [(str(o) +"apostrophe", 0)]

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def special_characters(word, o):
    if(isEnglish(word)):
        return [(str(o) +"special_characters", 0)]
    else:
        return [(str(o) +"special_characters", 1)]

def onlynum(word, o):
    if(word.isdigit()):
        return [(str(o) +"onlynum", 1)]
    else:
        return [(str(o) +"onlynum", 0)]

def contains_num(word, o):
    if(any(char.isdigit() for char in word)):
        return [(str(o) +"contains_num", 1)]
    else:
        return [(str(o) +"contains_num", 0)]

def ending_fullstop(word, o):
    if(word[-1] == '.'):
        return [(str(o) +"fullstop", 1)]
    else:
        return [(str(o) +"fullstop", 0)]

def minlen(word, o):
    if(len(word)>=2):
        return [(str(o) +"minlen", 1)]
    else:
        return [(str(o) +"minlen", 0)]

def punctuation(word, o):
    for i in word: 
      if i in string.punctuation: 
        return [(str(o) +"punctuation", 1)]
    return [(str(o) +"punctuation", 0)]

def all_punctuation(word, o):
    count = 0
    for i in word: 
      if i in string.punctuation: 
        count = count +1
    if(count == len(word)):
        return [(str(o) +"punctuation", 1)]
    else:
        return [(str(o) +"punctuation", 0)]

def is_stopword(word, o):
    stop_words = set(stopwords.words('spanish'))
    if(word in stop_words):
        return([(str(o) +"is_stop", 1)])
    else:
        return([(str(o) +"is_stop", 0)])


def isRomanNumeral(word, o):
    numeral = word.upper()
    validRomanNumerals = ["M", "D", "C", "L", "X", "V", "I"]
    for letters in numeral:
        if letters not in validRomanNumerals:
            return ([(str(o) +"is_roman", 0)])

    return ([(str(o) + "is_roman", 1)])


def contains_dots(word, o):
    if word.find('.')==-1:
        return ([(str(o) + "has_dot", 0)])

    return ([(str(o) + "has_dot", 1)])


def single_char(word, o):
    if len(word)==1:
        return ([(str(o) + "is_char", 1)])

    return ([(str(o) + "is_char", 0)])

def is_url(word, o):
    if re.match(regex, word) is not None:
        return ([(str(o) + "is_url", 1)])
    return ([(str(o) + "is_url", 0)])


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-4,-3, -2,-1,0,1,2, 3, 4]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            tag = sent[i+o][1]
            featlist = getfeats(word, o)
            features.extend(featlist)
            featlist = gettag(tag, o)
            features.extend(featlist)

            featlist = gethyphen(word, o)
            features.extend(featlist)

            featlist = capletter(word, o)
            features.extend(featlist)

            featlist = noun_suffix(word, o)
            features.extend(featlist)

            featlist = get_wordshape(word, o)
            features.extend(featlist)

            featlist = all_upper(word, o)
            features.extend(featlist)

            featlist = all_lower(word, o)
            features.extend(featlist)

            featlist = has_apostrophe(word, o)
            features.extend(featlist)

            featlist = special_characters(word, o)
            features.extend(featlist)

            featlist = onlynum(word, o)
            features.extend(featlist)

            featlist = contains_num(word, o)
            features.extend(featlist)

            featlist = ending_fullstop(word, o)
            features.extend(featlist)

            featlist = isRomanNumeral(word, o)
            features.extend(featlist)

            featlist = contains_dots(word, o)
            features.extend(featlist)

            featlist = single_char(word, o)
            features.extend(featlist)

            featlist = is_url(word, o)
            features.extend(featlist)

    
    word = sent[i][0]
    tag = sent[i][1]

    features.extend([("word_lower", word.lower())])

    features.extend([("word_len", len(word))])

    if (i == 0):
        features.extend([("firstword", 1)])
    else:
        features.extend([("firstword", 0)])

    features.extend([("bias", 1)])
    
    return dict(features)

if __name__ == "__main__":
    train = ConllCorpusReader('', 'training_data', ['words', 'pos', 'ignore', 'chunk'])
    dev = ConllCorpusReader('', 'validation_data', ['words', 'pos', 'ignore', 'chunk'])
    test = ConllCorpusReader('', 'testing_data', ['words', 'pos', 'ignore', 'chunk'])

    train_sents = list(train.iob_sents())
    dev_sents = list(dev.iob_sents())
    test_sents = list(test.iob_sents())

    # train_sents = list(conll2002.iob_sents('esp.train'))
    # dev_sents = list(conll2002.iob_sents('esp.testa'))
    # test_sents = list(conll2002.iob_sents('esp.testb'))


    X_train = []
    y_train = []

    train_feats = []
    train_labels = []

    for sent in train_sents:
        train_feats = []
        train_labels = []
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])
        X_train.append(train_feats)
        y_train.append(train_labels)

    X_test = []
    y_test = []

    test_feats = []
    test_labels = []

    for sent in test_sents:
        test_feats = []
        test_labels = []
        for i in range(len(sent)):
            feats = word2features(sent, i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])
        X_test.append(test_feats)
        y_test.append(test_labels)


    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    true_y_test = []
    for li in y_test:
        for label in li:
            true_y_test.append(label)
    true_y_test=np.array(true_y_test)

    samples = []
    item = 0
    for i in range(0, 120):
        item += 20
        samples.append(item)
    for i in range(0, 62):
        item += 200
        samples.append(item)
    samples.append(len(train_sents))


    all_f1_scores = []

    for no_of_samples in samples:
        model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.5,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        #random_sample_indices = np.random.randint(0, len(y_train), size=no_of_samples)

        lengths = np.array([len(x) for x in train_sents])
        ind = np.argpartition(lengths, -no_of_samples)[-no_of_samples:]
        for index in ind:
            for i in range(0, len(train_sents[index])):
                print(train_sents[index][i][0], end = " ")
            print()
        model.fit(X_train[ind], y_train[ind])
        y_pred = model.predict(X_test)

        true_y_pred = []
        for li in y_pred:
            for label in li:
                true_y_pred.append(label)
        true_y_pred=np.array(true_y_pred) 
        

        
        f1_score_of_test = f1_score(true_y_test, true_y_pred, average="macro")
        print("No of samples = ", no_of_samples, " f1 score = ", f1_score_of_test)
        all_f1_scores.append(f1_score_of_test)

    plt.plot(samples, all_f1_scores)
    plt.show()

    with open("length_baseline_english_results.txt", "w") as f:
        for i in range(len(samples)):
            f.write(str(samples[i])+" "+str(all_f1_scores[i])+"\n")