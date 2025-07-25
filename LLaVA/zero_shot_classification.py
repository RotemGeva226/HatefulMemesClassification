if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
    from LLaVAMemesDataset import LLaVAMemesDataset
    from LLaVA import LLaVA
    from utils import collate_fn
    from config import Config

    config_wrapper = Config()

    # Prepare dataframe
    df_test = pd.read_json(config_wrapper.config["test_path"], lines=True)

    # Load test dataset
    test_dataset = LLaVAMemesDataset(df_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config_wrapper.config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config_wrapper.config["num_workers"],
        collate_fn=collate_fn
    )

    # Load model
    model = LLaVA(
        model_id=config_wrapper.config["model_id"],
        device=config_wrapper.config["device"]
    )

    y_true = []
    y_pred = []
    for idx, image, prompt, label in tqdm(test_loader, desc="Samples"):
        response = model.generate_answer(image, prompt)
        response_clean = (response[0].split("ASSISTANT:")[-1].strip()).lower()
        predicted = 1 if "yes" in response_clean else 0

        y_true.append(label.cpu().numpy())
        y_pred.append(predicted)

    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))