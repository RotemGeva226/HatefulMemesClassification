import wandb

class WandbLogger:
    def __init__(self, config):
        print("Initializing WandbLogger")
        self.config = config
        self.run = wandb.init(
            project=config["project_name"],
            name=f"bs{config['batch_size']}-lr{config['learning_rate']}-{config['time']}",
            config=config
        )
        self.define_metrics()


    def define_metrics(self):
        print("Defining metrics")
        # Epoch-based metrics
        self.run.define_metric(step_metric="epoch", name="train_loss")
        for metric in ["loss", "auroc", "accuracy", "f1"]:
            self.run.define_metric(step_metric="epoch", name=f"val_{metric}")

        # Batch-based metrics
        self.run.define_metric(step_metric="batch_number", name="batch_loss")

    def log_epoch_metrics(self, epoch, loss, metrics, prefix="train"):
        log_dict = {
            "epoch": epoch + 1,
            f"{prefix}_loss": round(loss, 4),
        }

        if metrics:
            if "auroc" in metrics:
                log_dict[f"{prefix}_auroc"] = round(metrics["auroc"], 4)
            if "accuracy" in metrics:
                log_dict[f"{prefix}_accuracy"] = round(metrics["accuracy"], 4)
            if "f1" in metrics:
                log_dict[f"{prefix}_f1"] = round(metrics["f1"], 4)

        print(log_dict)
        self.run.log(log_dict)

    def log_batch_loss(self, loss, batch_number):
        self.run.log({
            "batch_number": batch_number,
            "batch_loss": round(loss,4)
        })

    @staticmethod
    def set_summary(key, value):
        wandb.run.summary[key] = value