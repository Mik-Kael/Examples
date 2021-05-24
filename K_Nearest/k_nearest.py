import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    data = pd.read_csv("car.data")
    columns = list(data.columns)
    code = []
    label_encoder = preprocessing.LabelEncoder()
    for i in columns:
        code.append(label_encoder.fit_transform(list(data[i])))
    predict = "class"
    x = list(zip(code[0], code[1], code[2], code[3], code[4], code[5]))
    y = list(code[-1])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    predicted = model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]
    for i in range(len(predicted)):
        print("Predicted:", names[predicted[i]], "Data:", x_test[i], "Actual:", names[y_test[i]])
    pass
