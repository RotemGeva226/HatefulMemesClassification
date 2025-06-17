import wandb

class WandbLogger:
    def __init__(self, config):
        self.config = config
        self.run = wandb.init(
            project=config["project_name"],
            name=f"bs{config['batch_size']}-lr{config['learning_rate']}-{config['time']}",
            config=config
        )
        self.define_metrics()


    def define_metrics(self):
        # Epoch-based metrics
        for metric in ["loss", "auroc", "accuracy", "f1"]:
            self.run.define_metric(step_metric="epoch", name=f"train_{metric}")
            self.run.define_metric(step_metric="epoch", name=f"val_{metric}")

        # Batch-based metrics
        self.run.define_metric(step_metric="batch_number", name="batch_loss")

    def log_epoch_metrics(self, epoch, loss, metrics, prefix="train"):
        log_dict = {
            "epoch": epoch + 1,
            f"{prefix}_loss": round(loss,4),
            f"{prefix}_auroc": round(metrics["auroc"],4),
            f"{prefix}_accuracy": round(metrics["accuracy"],4),
            f"{prefix}_f1": round(metrics["f1"],4)
        }
        self.run.log(log_dict)

    def log_batch_loss(self, loss, batch_number):
        self.run.log({
            "batch_number": batch_number,
            "batch_loss": round(loss,4)
        })

    @staticmethod
    def set_summary(key, value):
        wandb.run.summary[key] = value