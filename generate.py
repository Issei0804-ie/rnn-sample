import tensorflow as tf
import temp

def generate_text(model, start_string, word_to_id, id_to_word):
    # 評価ステップ（学習済みモデルを使ったテキスト生成）

    # 生成する文字数
    num_generate = 500

    # 開始文字列を数値に変換（ベクトル化）
    input_eval = [word_to_id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 結果を保存する空文字列
    text_generated = []

    # 低い temperature　は、より予測しやすいテキストをもたらし
    # 高い temperature は、より意外なテキストをもたらす
    # 実験により最適な設定を見つけること
    temperature = 0.5

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
    checkpoint_dir = './training_checkpoints'

    corpus, word2id, id2word = temp.load_dataset()

    model = temp.make_model(len(word2id), 64,64,1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))
    model.summary()

    print(generate_text(model, "日本", word2id,id2word))

if __name__ == "__main__":
    main()

