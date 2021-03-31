import csv
import numpy as np
import math
from gensim.models import Word2Vec
from nltk.corpus import stopwords

stopWords = set(stopwords.words("English"))

def read_data(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        stories = []
        for line in lines:
            paragraph = []
            paragraph.append(line[0])
            paragraph.append(line[1])
            paragraph.append(line[2])
            paragraph.append(line[3])
            paragraph.append(line[4])
            paragraph.append(line[5])
            paragraph.append(line[6])
            paragraph.append(line[7])
            stories.append(paragraph)
    return stories

def create_embedding(stories):
    sentences = []
    for paragraph in stories:
        for i in range(1, 7):
            sentences.append(paragraph[i].split())
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    return model

def evaluate(stories, model):
    correct = 0
    totals = 0
    for paragraph in stories:
        label = int(paragraph[7])
        avgVec = np.zeros(100)
        fifthVec = np.zeros(100)
        sixthVec = np.zeros(100)
        total = 0
        for i in range(1, 5):
            for word in paragraph[i]:
                if word not in stopWords:
                    try:
                        avgVec += model.wv[word]
                    except KeyError:
                        pass
                    total += 1
        avgVec = avgVec / total
        for word in paragraph[5]:
            if word not in stopWords:
                try:
                    fifthVec += model.wv[word]
                except KeyError:
                    pass
        fifthVec = fifthVec / len(paragraph[5])
        for word in paragraph[6]:
            if word not in stopWords:
                try:
                    sixthVec += model.wv[word]
                except KeyError:
                    pass
        sixthVec = sixthVec / len(paragraph[6])
        #diffOne = np.linalg.norm(fifthVec - avgVec)
        #diffTwo = np.linalg.norm(sixthVec - avgVec)
        
        try:
            cos_sim_one = np.dot(avgVec, fifthVec) / math.exp(math.log(np.linalg.norm(avgVec)) + math.log(np.linalg.norm(fifthVec)))
            cos_sim_two = np.dot(avgVec, sixthVec) / math.exp(math.log(np.linalg.norm(avgVec)) + math.log(np.linalg.norm(sixthVec)))
        except ValueError:
            cos_sim_one = np.dot(avgVec, fifthVec) / 0.00001
            cos_sim_two = np.dot(avgVec, sixthVec) / 0.00001
        
        if cos_sim_one < cos_sim_two:
            prediction = 1
            #print("prediction = " + str(prediction) + ", label = " + str(label))
        else:
            prediction = 2
            #print("prediction = " + str(prediction) + ", label = " + str(label))
        correct += int(prediction == label)
        totals += 1
    acc = correct / totals
    print("Accuracy =", acc)
            

def predict(stories, model):
    with open("prediction.csv", mode="w", newline='') as f:
        prediction_writer = csv.writer(f, delimiter=',', quotechar='"')
        prediction_writer.writerow(["Id", "Prediction"])
        for paragraph in stories:
            Id = paragraph[0]
            avgVec = np.zeros(100)
            fifthVec = np.zeros(100)
            sixthVec = np.zeros(100)
            total = 0
            for i in range(1, 5):
                for word in paragraph[i]:
                    if word not in stopWords:
                        try:
                            avgVec += model.wv[word]
                        except KeyError:
                            pass
                        total += 1
            avgVec = avgVec / total
            for word in paragraph[5]:
                if word not in stopWords:
                    try:
                        fifthVec += model.wv[word]
                    except KeyError:
                        pass
            fifthVec = fifthVec / len(paragraph[5])
            for word in paragraph[6]:
                if word not in stopWords:
                    try:
                        sixthVec += model.wv[word]
                    except KeyError:
                        pass
            sixthVec = sixthVec / len(paragraph[6])
            try:
                cos_sim_one = np.dot(avgVec, fifthVec) / math.exp(math.log(np.linalg.norm(avgVec)) + math.log(np.linalg.norm(fifthVec)))
                cos_sim_two = np.dot(avgVec, sixthVec) / math.exp(math.log(np.linalg.norm(avgVec)) + math.log(np.linalg.norm(sixthVec)))
            except ValueError:
                cos_sim_one = np.dot(avgVec, fifthVec) / 0.00001
                cos_sim_two = np.dot(avgVec, sixthVec) / 0.00001
            #len1 = len(paragraph[5])
            #len2 = len(paragraph[6])
            if cos_sim_one > cos_sim_two:
                prediction_writer.writerow([Id, 1])
            else:
                prediction_writer.writerow([Id, 2])

val_data = read_data("dev.csv")
val_model = create_embedding(val_data)
evaluate(val_data, val_model)
#test_data = read_data("test.csv")
#test_model = create_embedding(test_data)
#predict(test_data, test_model)