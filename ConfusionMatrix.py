import torch
#import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def ConfusionMatrix(model, dataloader, DEVICE):
    model.eval()

    true_label = []
    preds = []

    with torch.inference_mode():
        for image_batch, label_batch in dataloader:
            label_batch = label_batch.tolist()
            for i in label_batch:
                true_label.append(i)

            image_batch = image_batch.to(DEVICE)
            output = model(image_batch)
            output = output.tolist()

            for i in output:
                single_preds = np.argmax(i)
                preds.append(single_preds)

    accuracy = accuracy_score(true_label, preds)
    cm = confusion_matrix(true_label, preds)
    #sns.heatmap(cm, annot=True, xticklabels=)
    plt.matshow(A=cm, cmap=plt.cm.Blues)

    # draw matrix
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j,i], xy=(i,j),horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted lable')

    plt.xticks(range(0,10), labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    plt.yticks(range(0,10), labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])