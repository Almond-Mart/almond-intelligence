python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/koch.yaml \
    --fps 30 \
    --root data \
    --repo-id shawnptl8/koch_test_training \
    --tags tutorial \
    --warmup-time-s 10 \
    --episode-time-s 15 \
    --reset-time-s 15 \
    --num-episodes 50