import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib import style
from sklearn import linear_model

if __name__ == '__main__':
    data = pd.read_csv("student-mat.csv", ";", usecols=["G1", "G2", "G3", "studytime", "failures", "absences"])
    predict = "G3"
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    best = 0
    for i in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        if acc > best:
            best = acc
            with open("student_grade_model.pickle", "wb") as file:
                pickle.dump(linear, file)

    with open("student_grade_model.pickle", "rb") as file:
        linear_loaded = pickle.load(file)

    acc = linear_loaded.score(x_test, y_test)
    print("accuracy from pickle:", acc)

    predictions = linear.predict(x_test)
    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

    p = "G1"
    style.use("ggplot")
    plt.scatter(data[p], data[predict])
    plt.xlabel(p)
    plt.ylabel(predict)
    plt.show()
    pass
