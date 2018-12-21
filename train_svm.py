import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools


#Taken from scikit website
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def trainSVM():
    
    #Load our training and test descriptors
    descriptors = pickle.load(open("p1_training.dat", "rb"))
    tests = pickle.load(open("p1_testing.dat", "rb"))
    
    #Intialize SVM, with C value of 3
    SVM = LinearSVC(C=3)
    
    """
    Intialize labels for descriptors, each classifier has it's own value
    Grass = 1, Ocean = 2, Red Carpet = 3, Road = 4, Wheat Field = 5
    """
    labels = ([1] * 800) + ([2] * 800) + ([3] * 800) + ([4] * 800) + ([5] * 800)
    
    test_labels = ([1] * 200) + ([2] * 200) + ([3] * 200) + ([4] * 200) + ([5] * 200)
    #Fit the model according to training data
    SVM.fit(descriptors, labels)
    
    training_predictions = SVM.predict(descriptors)
    print("Training accuracy: %0.2f" % accuracy_score(labels, training_predictions))
    
    test_predictions = SVM.predict(tests)
    print("Testing accuracy: %0.2f" % accuracy_score(test_labels, test_predictions))
        
    cnf_matrix = confusion_matrix(test_predictions, test_labels)
    np.set_printoptions(precision=2)
    
    classe_names = ['grass', 'ocean', 'redcarpet', 'road', 'wheatfield']
    plot_confusion_matrix(cnf_matrix, classes=classe_names,
                      title='Confusion matrix, without normalization')
    
    plt.savefig("confusion.png")
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classe_names, normalize=True,
                      title='Normalized confusion matrix')
    plt.figure()
    plt.show()
    
    
if __name__ == "__main__":
    trainSVM()