SCRIPT = $1

if [ "$SCRIPT" == "eval" ]; then
    REPO = $2
    MODEL = $3

    python lerobot/scripts/control_robot.py record \
        --robot-path lerobot/configs/robot/koch.yaml \
        --fps 30 \
        --root data \
        --repo-id shawnptl8/eval_$REPO \
        --tags tutorial eval \
        --warmup-time-s 10 \
        --episode-time-s 15 \
        --reset-time-s 15 \
        --num-episodes 10 \
        --policy-overrides device=mps \
        -p outputs/train/$MODEL
elif [ "$SCRIPT" == "train" ]; then
    USERNAME = $2
    DATASET = $3

    DATA_DIR=data python lerobot/scripts/train.py \
        dataset_repo_id=$USERNAME/$DATASET \
        policy=act_koch_real \
        env=koch_real \
        hydra.run.dir=outputs/train/act_$DATASET \
        hydra.job.name=act_$DATASET \
        device=cuda \
        wandb.enable=true
elif [ "$SCRIPT" == "record" ]; then
    REPO = $2

    python lerobot/scripts/control_robot.py record \
        --robot-path lerobot/configs/robot/koch.yaml \
        --fps 30 \
        --root data \
        --repo-id shawnptl8/$REPO \
        --warmup-time-s 10 \
        --episode-time-s 20 \
        --reset-time-s 15 \
        --num-episodes 50
elif [ "$SCRIPT" == "teleop" ]; then
    python lerobot/scripts/control_robot.py teleoperate \
        --robot-path lerobot/configs/robot/koch.yaml
elif [ "$SCRIPT" == "visualize" ]; then
    REPO = $2

    python lerobot/scripts/visualize_dataset_html.py \
        --root data \
        --repo-id shawnptl8/$REPO
fi