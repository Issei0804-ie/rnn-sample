import csv
import os

import spacy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    #data = data

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

    num = len(dataset)
    validation = dataset.take(int(num*0.2))
    test = dataset.take(int(num*0.2))
    train = dataset
    return train, validation, test


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def save_graph(path: str, history ):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("sample")
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'Val_loss'], loc='upper left')
    plt.savefig(path)
    plt.clf()



def generate_text(model, start_string, word_to_id, id_to_word):
  # 評価ステップ（学習済みモデルを使ったテキスト生成）

  # 生成する文字数
  num_generate = 1000

  # 開始文字列を数値に変換（ベクトル化）
  input_eval = [word_to_id[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 結果を保存する空文字列
  text_generated = []

  # 低い temperature　は、より予測しやすいテキストをもたらし
  # 高い temperature は、より意外なテキストをもたらす
  # 実験により最適な設定を見つけること
  temperature = 1.0

  # ここではバッチサイズ　== 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # バッチの次元を削除
      predictions = tf.squeeze(predictions, 0)

      # カテゴリー分布をつかってモデルから返された文字を予測
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 過去の隠れ状態とともに予測された文字をモデルへのつぎの入力として渡す
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(id_to_word[predicted_id])

  return (start_string + ''.join(text_generated))

def main():
    corpus, word_to_id, id_to_word = load_dataset()

    train, validation, test = make_batch(corpus, 64)

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

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(1e-4))

    # チェックポイントが保存されるディレクトリ
    checkpoint_dir = './training_checkpoints'
    # チェックポイントファイルの名称
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(train, epochs=100, validation_data=validation, verbose=1, callbacks=[checkpoint_callback])
    save_graph("output/nn.jpg", history)

if '__main__' == __name__:
    main()
