# cd /home/ubuntu/almond-intelligence
# conda activate lerobot

DATA_DIR=data python lerobot/scripts/train.py \
    dataset_repo_id=shawnptl8/koch_test_training \
    policy=act_koch_real \
    env=koch_real \
    hydra.run.dir=outputs/train/act_koch_test_training \
    hydra.job.name=act_koch_test_training \
    device=cpu \
    wandb.enable=true