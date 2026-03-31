#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kaixin/yisong/demo"
QURA_ROOT="$ROOT/third_party/qura/ours/main"
PYTHON="/home/kaixin/anaconda3/envs/demo_adv/bin/python"
LOG_DIR="$ROOT/outputs/tiny_retest/logs"

MODE="${1:-all}"            # train | eval | all
GPU="${GPU:-3}"
MODELS="${MODELS:-vit resnet18 vgg16}"

mkdir -p "$LOG_DIR"

run_train() {
  local model="$1"
  local train_log="$LOG_DIR/${model}_tiny_train.log"
  local extra_args=()

  case "$model" in
    vit)
      extra_args=(--l_r 1e-4 --epochs 50 --batch_size 64 --num_workers 4 --warmup_epochs 5)
      ;;
    resnet18)
      extra_args=(--l_r 1e-2 --epochs 100 --batch_size 128 --num_workers 8)
      ;;
    vgg16)
      extra_args=(--l_r 1e-2 --epochs 100 --batch_size 128 --num_workers 8)
      ;;
    *)
      echo "Unsupported model: $model" >&2
      exit 1
      ;;
  esac

  echo "[train] $model -> $train_log"
  (
    cd "$QURA_ROOT/setting"
    "$PYTHON" -u train_model.py \
      --model "$model" \
      --dataset tiny_imagenet \
      --gpu "$GPU" \
      "${extra_args[@]}"
  ) | tee "$train_log"
}

run_eval() {
  local model="$1"
  local config="$2"
  local tag="$3"
  local eval_log="$LOG_DIR/${model}_${tag}.log"

  echo "[eval] $model / $config -> $eval_log"
  (
    cd "$QURA_ROOT"
    "$PYTHON" -u main.py \
      --config "$config" \
      --model "$model" \
      --dataset tiny_imagenet \
      --type bd \
      --enhance 1 \
      --gpu "$GPU"
  ) | tee "$eval_log"
}

for model in $MODELS; do
  if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    run_train "$model"
  fi

  if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
    case "$model" in
      vit)
        run_eval "$model" "configs/cv_vit_tiny_8_8_bd.yaml" "tiny_bd_8_8"
        run_eval "$model" "configs/cv_vit_tiny_4_8_bd.yaml" "tiny_bd_4_8"
        ;;
      resnet18|vgg16)
        run_eval "$model" "configs/cv_tiny_8_8_bd.yaml" "tiny_bd_8_8"
        run_eval "$model" "configs/cv_tiny_4_4_bd.yaml" "tiny_bd_4_4"
        ;;
      *)
        echo "Unsupported model: $model" >&2
        exit 1
        ;;
    esac
  fi
done
