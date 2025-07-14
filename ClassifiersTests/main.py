from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ClassifiersTests.models_factory import create_classifier
from ClassifiersTests.classifiers_config import classifier_config
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils import compute_sample_weight



def load_embeddings_dataframe(directory_path):
    csv_file = [file_path for file_path in Path(directory_path).rglob("*.csv")][0]
    df_embeddings = pd.read_csv(csv_file)

    embeddings = np.stack([np.load(path) for path in df_embeddings['embedding_path']])

    # Create a DataFrame for the embeddings
    embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])

    # Concatenate with the original id and label columns
    new_df = pd.concat([df_embeddings['label'].reset_index(drop=True), embedding_df], axis=1)
    y = new_df.pop('label')
    return new_df, y

def run_experiment():
    # Load embeddings from CSV files
    train_embeddings_path = classifier_config.embeddings.train_path
    val_embeddings_path = classifier_config.embeddings.validation_path
    test_embeddings_path = classifier_config.embeddings.test_path

    x_train, y_train = load_embeddings_dataframe(train_embeddings_path)
    x_validation, y_validation = load_embeddings_dataframe(val_embeddings_path)
    x_test, y_test = load_embeddings_dataframe(test_embeddings_path)

    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)

    # Train the classifier
    classifier = create_classifier(classifier_config.classifier.name,classifier_config.classifier.params)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    classifier.fit(x_train,y_train, sample_weight)
    print("Training complete.")

    # Get predictions and probabilities
    y_val_proba = classifier.predict_proba(x_validation)[:, 1]
    y_test_proba = classifier.predict_proba(x_test)[:, 1]
    y_val_preds = classifier.predict(x_validation)
    y_test_pred = classifier.predict(x_test)
    print("Validation:")
    print(classification_report(y_validation, y_val_preds))
    print("AUROC:", roc_auc_score(y_validation, y_val_proba))
    print("Test:")
    print(classification_report(y_test, y_test_pred))
    print("AUROC:", roc_auc_score(y_test, y_test_proba))

if __name__ == "__main__":
    run_experiment()