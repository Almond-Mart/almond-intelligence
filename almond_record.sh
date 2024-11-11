python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/koch.yaml \
    --fps 30 \
    --root data \
    --repo-id shawnptl8/koch_almond_pick_box \
    --tags demo \
    --warmup-time-s 10 \
    --episode-time-s 20 \
    --reset-time-s 15 \
    --num-episodes 50