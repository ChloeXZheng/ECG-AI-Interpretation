import src.train as tf

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# creating a confusion matrix (basically like a graph of actual valence vs predicted valence)
ConfusionMatrixDisplay.from_predictions(
    tf.y_test,
    tf.y_pred,
    labels=[1, 2, 3, 4, 5]
)

plt.xlabel("Predicted Valence")
plt.ylabel("Actual Valence")
plt.title("Random Forest Confusion Matrix")
plt.show()