from tensorflow import keras


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def review_encode(string):
    encoded = [1]
    for word in string:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


if __name__ == "__main__":
    data = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)
    word_index = data.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    max_length = max((len(i) for i in train_data))
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                            maxlen=max_length)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                           maxlen=max_length)
    model = keras.models.load_model("model.h5")

    with open("review.txt", encoding="utf-8") as file:
        for line in file.readlines():
            new_line = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "")
            encode = review_encode(new_line)
            encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=max_length)
            predict = model.predict([encode])
            print(line)
            print(encode)
            print(predict[0])
    print(test_labels)
    print(decode_review(test_data[1]))
    predict = model.predict([test_data[2]])
    print(predict[0])
    prediction = model.predict(test_data[:5])
    for i in range(5):
        print("Review:", decode_review(test_data[i]))
        print("Prediction:", prediction[i])
        print("Actual:", test_labels[i])
