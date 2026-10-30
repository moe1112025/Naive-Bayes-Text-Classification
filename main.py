# Import pandas to read the text dataset from a CSV file.
import pandas as pd  # Read and organize tabular data.

# Import matplotlib and seaborn to draw the confusion matrix.
import matplotlib.pyplot as plt  # Display charts.

import seaborn as sns  # Draw a readable confusion-matrix heatmap.
# Import the tools needed to split data and build a text classifier.
from sklearn.model_selection import train_test_split  # Keep test data separate.
from sklearn.feature_extraction.text import CountVectorizer  # Convert words into counts.
from sklearn.naive_bayes import MultinomialNB  # Use a text classifier.
from sklearn.pipeline import Pipeline  # Keep text conversion and classification together.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Measure predictions.

# Define one function so the project can be run from a single entry point.
def main():

    # Read the CSV file; it must contain columns named 'text' and 'label'.
    data = pd.read_csv("synthetic_text_data.csv")

    # Stop early with a clear message if the required columns are missing.
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Remove rows without text or labels and convert both columns to strings.
    data = data.dropna(subset = ["text", "label"]).copy()
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)

    # Split the dataset without stratification to fit the small dummy dataset size.
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size = 0.2, random_state = 42
    )

    # CountVectorizer converts text into word-count features before Naive Bayes sees it.
    model = Pipeline([
        ("word_counts", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Learn vocabulary and class patterns from training data only.
    model.fit(X_train, y_train)

    # Predict the labels for the held-out test messages.
    predictions = model.predict(X_test)

    # Print accuracy and per-class precision, recall, and F1-score.
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division = 0))

    # Use a stable label order for the confusion matrix axes.
    labels = sorted(data["label"].unique())
    matrix = confusion_matrix(y_test, predictions, labels = labels)

    # Display correct and incorrect predictions in a heatmap.
    sns.heatmap(matrix, annot = True, fmt = "d", cmap = "Blues", xticklabels = labels, yticklabels = labels)

    plt.title("Naive Bayes Text Classification")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    # Predict a new message using the same fitted pipeline.
    new_text = ["I enjoy learning artificial intelligence and machine learning"]

    print(f"\nExample message: {new_text[0]}")
    print(f"Predicted category: {model.predict(new_text)[0]}")

# Run the project only when this file is executed directly.
if __name__ == "__main__":
    main()
