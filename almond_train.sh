USERNAME = $1
DATASET = $2

DATA_DIR=data python lerobot/scripts/train.py \
    dataset_repo_id=$USERNAME/$DATASET \
    policy=act_koch_real \
    env=koch_real \
    hydra.run.dir=outputs/train/act_$DATASET \
    hydra.job.name=act_$DATASET \
    device=cuda \
    wandb.enable=true