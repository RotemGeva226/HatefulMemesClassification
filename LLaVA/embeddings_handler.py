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
    import os
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA


    def check_similarity(x, pca=False):
        """Wide std
        Mean needs to be close to 0, std close to 1.
        High similarity count should be much low then: n(n-1)/2
        """
        similarity_matrix = cosine_similarity(x)

        # Set diagonal to 0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)

        # High similarity count (> 0.95 means almost identical)
        high_sim_count = np.sum(similarity_matrix > 0.95)
        print(f"Number of embedding pairs with cosine similarity > 0.95: {high_sim_count}")

        # Mean and std
        print("Mean embedding:", np.mean(x, axis=0)[:5])
        print("Std of embeddings:", np.std(x, axis=0)[:5])

        dists = cosine_distances(x)
        np.fill_diagonal(dists, np.nan)

        plt.hist(dists[~np.isnan(dists)].flatten(), bins=100)
        plt.title("Histogram of Pairwise Cosine Distances")
        plt.xlabel("Cosine Distance")
        plt.ylabel("Frequency")
        plt.show()

        if pca:
            pca = PCA()
            pca.fit(x)
            explained_variance_ratio = pca.explained_variance_ratio_

            plt.plot(np.cumsum(explained_variance_ratio))
            plt.title("Cumulative Explained Variance by PCA")
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Variance")
            plt.grid(True)
            plt.show()


    def analyze_embeddings(model, loader):
        all_embeddings = []
        all_labels = []
        for idx, images, prompts, labels in tqdm(loader, desc=f"Samples"):
            embeddings_pooled = model.generate_embeddings(images, prompts)
            # utils.save_embedding(embeddings_pooled, idx[0])
            all_embeddings.extend(embeddings_pooled.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        X = np.array(all_embeddings)
        y = np.array(all_labels)
        utils.plot_tsne(X,y, perplexity=15)
        utils.plot_umap(X,y)
        check_similarity(X)

    def create_embeddings_dataset(model, loader):
        embedding_dir = os.path.join(os.getcwd(), "llava_embeddings_mean_pool_last_hidden_test")
        rows = []
        for idx, images, prompts, labels in tqdm(loader, desc="Samples"):
            embeddings_pooled = model.generate_embeddings(images, prompts)
            embeddings_np = embeddings_pooled.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i, emb, label in zip(idx, embeddings_np, labels_np):
                embedding_path = os.path.join(embedding_dir, f"{i}.npy")
                np.save(embedding_path, emb)

                rows.append({
                    "id": i,
                    "label": label,
                    "embedding_path": embedding_path
                })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(embedding_dir, "test_embeddings.csv"), index=False)


    config_wrapper = Config()

    # Prepare dataframe
    df = pd.read_json(config_wrapper.config["train_path"], lines=True)

    # Load train dataset
    dataset = LLaVAMemesDataset(df)
    loader = DataLoader(
        dataset,
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

    analyze_embeddings(model, loader)
