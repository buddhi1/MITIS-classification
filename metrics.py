from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from utils import results
import json
from pathlib import Path

class Metrics():

    def __init__(self, classes):

        self.epoch = 0
        self.loss = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.train_time = 0
        self.train_pred_time = 0
        self.test_pred_time = 0

        self.classes = classes

    def update_epoch(self, epoch, loss, y_true, y_pred):

        self.loss = loss
        self.epoch = epoch

        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted')
        self.recall = recall_score(y_true, y_pred, average='weighted')
        self.f1_score = f1_score(y_true, y_pred, average='weighted')

    def update_training_time(self, time):
        self.train_time = time

    def update_pred_time(self, train_pred, test_pred):
        self.train_pred_time = train_pred
        self.test_pred_time = test_pred

    def save(self, params, path, name):
        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        log_file = open(f'{path}{name}.json', "w")


        json.dump({
            "params" : params,
            "metrics" : vars(self)
        }, log_file, indent=4)





