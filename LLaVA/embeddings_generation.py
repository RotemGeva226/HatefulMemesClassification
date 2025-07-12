if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    from LLaVAMemesDataset import LLaVAMemesDataset
    from LLaVA import LLaVA
    from utils import collate_fn
    from config import Config
    from tqdm import tqdm
    import utils
    import numpy as np

    config_wrapper = Config()

    # Prepare dataframe
    df_train = pd.read_json(config_wrapper.config["train_path"], lines=True)

    # Load train dataset
    train_dataset = LLaVAMemesDataset(df_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_wrapper.config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config_wrapper.config["num_workers"],
        collate_fn=collate_fn
    )

    # Init model
    model = LLaVA(
        model_id=config_wrapper.config["model_id"],
        device=config_wrapper.config["device"]
    )

    all_embeddings = []
    all_labels = []
    for idx, images, prompts, labels in tqdm(train_loader, desc=f"Train samples"):

        embeddings_pooled = model.generate_embeddings(images, prompts)
        utils.save_embedding(embeddings_pooled, idx[0])
        all_embeddings.extend(embeddings_pooled.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    utils.plot_tsne(X, y)
