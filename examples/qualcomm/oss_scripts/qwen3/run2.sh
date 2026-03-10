export PTE_DIR=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4/qwen_qnn
export QNN_SDK_ROOT=/opt/qnn
export EXECUTORCH_ROOT=/workspaces/tman/test/executorch
export BUILD_FOLD=/workspaces/tman/test/executorch/build-android 
export MODEL_DIR=/workspaces/tman/adb_test/adb
export TOKENIZER=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4/tokenizer.json
export OUT_DIR=/workspaces/tman/adb_test/adb
adb push /workspaces/tman/test/executorch/pcre2/build-android/libpcre2-8.so /data/local/tmp/root/executorch/single_llama/
python examples/qualcomm/oss_scripts/qwen3/qwen3.py \
  -b build-android \
  -s emulator-5554 \
  -m SM8650 \
  --ptq 16a4w \
  --model_dir ${MODEL_DIR} \
  --tokenizer_model ${TOKENIZER} \
  --model_mode kv \
  --max_seq_len 2000 \
  --prompt "hello what is apple" \
  --llama_model qwen3 \
  --use_tman \
  --num_sharding 1 \
  --kv_updater shift_pointer \
  --pre_gen_pte ${PTE_DIR}  \
  --build_folder ${BUILD_FOLD} \
    --device "a50cde57"\
	--artifact ${OUT_DIR} 