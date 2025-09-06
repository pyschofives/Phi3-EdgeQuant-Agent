import os
import wandb

if os.getenv("WANDB_DISABLED", "false").lower() != "true":
    project = os.getenv("WANDB_PROJECT", "Phi3-EdgeQuant-Agent")
    entity = os.getenv("WANDB_ENTITY", None)
    wandb_dir = os.getenv("WANDB_DIR", "./wandb_logs")

    wandb.init(project=project, entity=entity, dir=wandb_dir)
