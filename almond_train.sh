DATA_DIR=data python lerobot/scripts/train.py \
    dataset_repo_id=shawnptl8/koch_almond_pick_box \
    policy=act_koch_real \
    env=koch_real \
    hydra.run.dir=outputs/train/act_koch_almond_pick_box \
    hydra.job.name=act_koch_almond_pick_box \
    device=cuda \
    wandb.enable=true