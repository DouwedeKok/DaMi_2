from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest


def load_files_folds(path, categories):
    return load_files(path, categories=categories)


def create_vectorizer_unigram():
    return CountVectorizer()


def create_vectorizer_bigram():
    return CountVectorizer(analyzer='word', ngram_range=(2, 2))


def create_feature_counts(vec, corpus):
    return vec.fit_transform(corpus)


def create_bigram_feature_counts(vec, corpus):
    return vec.fit_transform(corpus)


def create_features(vec):
    return vec.get_feature_names_out()


def extract_features_percentage(X, y, per):
    return SelectPercentile(chi2, percentile=per).fit_transform(X, y)


def extract_features_number(X, y, num):
    return SelectKBest(chi2, k=num).fit_transform(X, y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_dec = "Data/op_spam_v1.4/negative_polarity/deceptive_from_MTurk"
    path_truth = "Data/op_spam_v1.4/negative_polarity/truthful_from_Web"
    corpus_dec_train = load_files_folds(path_dec, ['fold1', 'fold2', 'fold3', 'fold4'])
    corpus_dec_test = load_files_folds(path_dec, ['fold5'])
    corpus_truth_train = load_files_folds(path_truth, ['fold1', 'fold2', 'fold3', 'fold4'])
    corpus_truth_test = load_files_folds(path_truth, ['fold5'])

    vec_uni = create_vectorizer_unigram()
    create_feature_counts(vec_uni, corpus_dec_train)
    create_features(vec_uni)

    vec_bi = create_vectorizer_bigram()
    create_feature_counts(vec_bi, corpus_dec_train)
    create_features(vec_bi)

print('ho')
