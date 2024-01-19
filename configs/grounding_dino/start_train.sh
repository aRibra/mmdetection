# start_train.sh
EPOCHS=2
LR=0.00005
BATCH_SIZE=4
GPU=0
LOGFILE=LOG.mini_coco_E1
SCREEN_SESSION_NAME=mini_coco_E1
EXP_ROOT_DIR=/mnt/disks/ext/exps/mini_coco/
WORK_DIR_NAME=E1
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py configs/grounding_dino/grounding_dino_swin-t_finetune_32xb1_1x_coco.py --cfg-options max_epoch='$EPOCHS' --cfg-options train_cfg.max_epochs='$EPOCHS'  --cfg-options param_scheduler.0.end='$EPOCHS' --cfg-options optim_wrapper.optimizer.lr='$LR' --cfg-options train_dataloader.batch_size='$BATCH_SIZE' --work-dir '$EXP_ROOT_DIR'/grounding_dino_swin-t_finetune_custom_dataset/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"












# DEBUG training start_train.sh
EPOCHS=1
LR=0.00005
BATCH_SIZE=4
GPU=0
LOGFILE=LOG.debug
SCREEN_SESSION_NAME=debug
EXP_ROOT_DIR=/mnt/disks/ext/exps/debug
WORK_DIR_NAME=debug_E
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py configs/grounding_dino/grounding_dino_swin-t_finetune_32xb1_1x_coco.py --cfg-options max_epoch='$EPOCHS' --cfg-options train_cfg.max_epochs='$EPOCHS'  --cfg-options param_scheduler.0.end='$EPOCHS' --cfg-options optim_wrapper.optimizer.lr='$LR' --cfg-options train_dataloader.batch_size='$BATCH_SIZE' --work-dir '$EXP_ROOT_DIR'/grounding_dino_swin-t_finetune_custom_dataset/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"