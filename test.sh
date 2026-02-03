set -e

mkdir -p trash

CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/flow_tiny/tiny_split1_10step_3refine_1rw_hmloss_config.py \
  work_dirs/1shot-swin-flow-tiny/split1_v1_10step_3refine_1rw_hmloss/best_PCK_epoch_200.pth \
  > trash/test_1rw.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python test.py \
  configs/flow_tiny/tiny_split1_10step_3refine_4gl_1rw_hmloss_config.py \
  work_dirs/1shot-swin-flow-tiny/split1_v1_10step_3refine_4gl_1rw_hmloss/best_PCK_epoch_200.pth \
  > trash/test_4gl.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/flow_tiny/tiny_split1_10step_3refine_6gl_1rw_hmloss_config.py \
  work_dirs/1shot-swin-flow-tiny/split1_v1_10step_3refine_6gl_1rw_hmloss/best_PCK_epoch_190.pth \
  > trash/test_6gl.log 2>&1 &

wait
echo "All tests finished."