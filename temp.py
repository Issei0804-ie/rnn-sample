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
    #data = data[:30]

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


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def make_batch(corpus: list, seq_length, test_data=0.2):
    BATCH_SIZE = 64
    char_dataset = tf.data.Dataset.from_tensor_slices(corpus)

    sequences = char_dataset.batch(10, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(1).batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def main():
    corpus, word_to_id, id_to_word = load_dataset()

    dataset = make_batch(corpus, 64)
    for input_example, target_example in dataset.take(1):
        print('Input data: ', input_example)
        print(input_example.shape)
        print('Target data:', target_example)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_to_id), 64, batch_input_shape=[64, None]),
        tf.keras.layers.GRU(64,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'
                            ),
        tf.keras.layers.Dense(len(word_to_id))
    ])
    model.summary()
    for input_example_batch, target_example_batch in dataset.take(2):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(1e-4))


    history = model.fit(dataset, epochs=1000)


if '__main__' == __name__:
    main()
