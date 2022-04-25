import csv
import spacy
import tensorflow as tf
import numpy as np


def make_dict(word: str, word_to_id: dict, id_to_word: dict):
    if word not in word_to_id:
        n = len(word_to_id)
        word_to_id[word] = n
        id_to_word[n] = word_to_id
    return word_to_id, id_to_word

def load_dataset():
    filename = "fakenews.csv"
    csv_file = open(filename, "r", encoding="utf-8", errors="", newline="")
    file = csv.reader(csv_file, delimiter=",")

    next(file)
    data = []
    for line in file:
        data.append(line[1])

    nlp = spacy.load('ja_ginza_electra')
    data = data[:3]

    word_to_id = {}
    id_to_word = {}
    corpus = []
    count = 0
    for datum in data:
        if count == 10684:
            continue
        print(datum)
        doc = nlp(datum)
        for tok in doc:
            word_to_id, id_to_word = make_dict(tok.text, word_to_id, id_to_word)
            corpus.append(word_to_id[tok.text])
        print("finish")
        count += 1

    print(len(word_to_id))
    print(word_to_id)
    print(corpus)
    return corpus, word_to_id, id_to_word


def make_batch(corpus:list, rnn_num:int, test_data = 0.2):
    BATCH_SIZE = len(corpus) - rnn_num + 1
    input = []
    target = []
    for i in range(BATCH_SIZE):
        input.append(corpus[i:rnn_num+i])
        target.append(corpus[(rnn_num+i)%len(corpus)])

    flatten_list = [v for child_list in input for v in child_list]
    input = np.array(flatten_list).reshape([-1,rnn_num])
    target = np.array(target).reshape([-1,1])
    return input, target



def main():
    corpus, word_to_id, id_to_word = load_dataset()

    input, target = make_batch(corpus, 64)
    train_data = tf.data.Dataset.from_tensor_slices((input, target)).shuffle(input.shape[0]).batch(64)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_to_id), 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10)



if '__main__' == __name__:
    main()