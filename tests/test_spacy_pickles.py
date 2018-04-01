import pickle

with open('../dataset/spacy_lemmas/train_lemmatized.pkl', 'rb') as f:
    mynewlist = pickle.load(f)

print(mynewlist)
