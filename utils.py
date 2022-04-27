
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20, 20

def results(x_true, x_pred, y_true, y_pred, classes, params, path=None, name=None):

    if path is None and name is None:
        path = f'models/{params["model_type"]}/{params["exp_name"]}/'
        name = f'{params["model_type"]}-{params["exp_name"]}'


    # Create folder
    Path(path).mkdir(parents=True, exist_ok=True)

    # Log
    log_file = open(f'{path}log.json', "w")
    json.dump(params, log_file, indent=4)

    # Train results
    x_pred_ = x_pred.argmax(dim=1)

    #classification report
    report = classification_report(x_true, x_pred_, target_names=classes,output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    accuracy_report = df_classification_report.tail(3)
    accuracy_report.to_csv(path+'train_accuracy_report.csv')
    df_classification_report.drop(df_classification_report.tail(3).index, inplace=True)
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    df_classification_report.to_csv(path+'train_classification_report.csv')

    # AUC curve
    x_true_ohe = np.zeros((len(x_pred), len(classes)))
    for idx, lbl in enumerate(x_true):
        x_true_ohe[idx][lbl] = 1

    x_pred = x_pred.detach().numpy()
    plot_multiclass_roc(x_true_ohe,x_pred, classes=classes, path=path, name='train-'+name)

    # Confusion matrix
    cm = confusion_matrix(x_true, x_pred_)
    plot_confusion_matrix(cm, classes, path=path, name='train-'+name)



    # Test results
    y_pred_ = y_pred.argmax(dim=1)

    #classification report
    report = classification_report(y_true, y_pred_, target_names=classes,output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    accuracy_report = df_classification_report.tail(3)
    accuracy_report.to_csv(path+'test-accuracy_report.csv')
    df_classification_report.drop(df_classification_report.tail(3).index, inplace=True)
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    df_classification_report.to_csv(path+'test-classification_report.csv')

    # AUC curve
    y_true_ohe = np.zeros((len(y_pred), len(classes)))
    for idx, lbl in enumerate(y_true):
        y_true_ohe[idx][lbl] = 1

    y_pred = y_pred.detach().numpy()
    plot_multiclass_roc(y_true_ohe,y_pred, classes=classes, path=path, name='test-'+name)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_)
    plot_confusion_matrix(cm, classes, path=path, name='test-'+name)

def get_color(idx):
    if idx < 10:
        return '#f500dc'
    elif idx < 20:
        return '#00f500'
    elif idx < 30:
        return '#00e0f5'
    elif idx < 40:
        return '#000cf5'
    elif idx < 50:
        return '#f5e900'
    elif idx < 60:
        return '#f58f00'
    else:
        return '#f50000'


def plot_multiclass_roc(y_true, y_pred, classes, path, name):
    n_classes = len(classes)
    lw=1

    items = []
    labels = ['item_id', 'fpr', 'tpr', 'roc_auc']

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        items.append((i, fpr, tpr, roc_auc))

    df = pd.DataFrame.from_records(items, columns=labels)
    df = df.sort_values(by=['roc_auc'], ascending=False)
    for idx, (_, row) in enumerate(df.iterrows()):
        color = get_color(idx)
        plt.plot(row['fpr'], row['tpr'], lw=lw, color=color,
                 label=f'{classes[row["item_id"]]} (area = {row["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic for {name}')
    plt.legend(loc='lower right',
              fancybox=True, shadow=True, ncol=3, prop={'size': 12})
    plt.savefig(f'{path}{name}-roc.png', bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_confusion_matrix(cm, classes, path, name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {name}')
    plt.savefig(f'{path}{name}-cm.png', bbox_inches='tight')
    plt.clf()
    plt.close()