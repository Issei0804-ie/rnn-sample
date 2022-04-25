import csv
import spacy


def make_dict(word: str, word_to_id: dict, id_to_word: dict):
    if word not in word_to_id:
        n = len(word_to_id)
        word_to_id[word] = n
        id_to_word[n] = word_to_id
    return word_to_id, id_to_word


filename = "fakenews.csv"
csv_file = open(filename, "r", encoding="utf-8", errors="", newline="")
file = csv.reader(csv_file, delimiter=",")

next(file)
data = []
for line in file:
    data.append(line[1])

nlp = spacy.load('ja_ginza_electra')
print(len(data))
print(data[10684])
print("******")
print(data[10685])
data = data[10680:10700]

word_to_id = {}
id_to_word = {}
corpus = []
count=0
for datum in data:
    if count==10684:
        continue
    print(datum)
    doc = nlp(datum)
    for tok in doc:
        word_to_id, id_to_word = make_dict(tok.text, word_to_id, id_to_word)
        corpus.append(word_to_id[tok.text])
    print("finish")
    count+=1

print(len(word_to_id))
print(word_to_id)
print(corpus)