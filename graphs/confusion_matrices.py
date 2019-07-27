import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

classes = ['agree', 'disagree', 'discuss', 'unrelated']

# Current: Logistic regresiion - improved system - test set
cm = [[95, 1, 100, 128], 
      [20, 21, 25, 23], 
      [68, 12, 652, 265], 
      [267, 4, 539, 2811]]


plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (test set)")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = 1610 / 2.
for i, j in itertools.product(range(4), range(4)):
    plt.text(j, i, cm[i][j],
             horizontalalignment="center",
             color="white" if cm[i][j] > thresh else "black")

plt.show()


