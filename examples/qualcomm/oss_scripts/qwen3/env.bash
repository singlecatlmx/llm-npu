export ANDROID_NDK_ROOT=/workspaces/tman/android/android-ndk-r26c
export QNN_SDK_ROOT=/workspaces/tman/qnn/qairt/2.37.0.250724
# export QNN_SDK_ROOT=/workspaces/tman/test/executorch/QNNv2.44.0.260225/qairt/2.44.0.260225
export QNN_INCLUDE=$QNN_SDK_ROOT/include/QNN
export HEXAGON_SDK_ROOT=/workspaces/tman/hexagon_sdk/hexagon
export EXECUTORCH_ROOT=/workspaces/tman/test/executorch
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH="/workspaces/tman/test/executorch:$PYTHONPATH"
export MODEL_DIR=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4
export OUT_DIR=/workspaces/tman/adb_test/adb
export GPTQMODEL_DISABLE_WARMUP=1
export DEVICE_DIR=/data/local/tmp/llama
export ADB_SERVER_SOCKET=tcp:host.docker.internal:5037
export PTE_DIR=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4/qwen_qnn
export QNN_OP_PACKAGE_PATHS="/workspaces/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/build/x86_64-linux-clang/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:CPU"
export HEXKL=/workspaces/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/tools/hexkl_addon

python examples/qualcomm/oss_scripts/qwen3/qwen3.py \
  -b build-android \
  -s emulator-5554 \
  -m SM8650 \
  --ptq 16a4w \
  --model_dir ${MODEL_DIR} \
  --tokenizer_model ${MODEL_DIR}/tokenizer.json \
  --model_mode hybrid \
  --max_seq_len 128 \
  --prefill_ar_len 32 \
  --prompt "Are you conscious?" \
  --llama_model qwen3 \
  --artifact ${MODEL_DIR}/qwen_qnn \
  --use_tman \
  --compile_only \
  --num_sharding 1 \
  --kv_updater shift_pointer \
  --verbose 2>&1 | tee qwen3_tman_hexagon_v75_final_output_hybrid.log

  python examples/qualcomm/oss_scripts/qwen3/qwen3_new.py \
  -b build-android \
  -s emulator-5554 \
  -m SM8650 \
  --ptq 16a4w \
  --model_dir ${MODEL_DIR} \
  --tokenizer_model ${MODEL_DIR}/tokenizer.json \
  --model_mode hybrid \
  --max_seq_len 128 \
  --prefill_ar_len 32 \
  --prompt "Are you conscious?" \
  --llama_model qwen3 \
  --artifact ${MODEL_DIR}/qwen_qnn_new \
  --use_tman \
  --compile_only \
  --num_sharding 1 \
  --kv_updater shift_pointer \
  --verbose 2>&1 | tee qwen3_tman_hexagon_v75_final_output_hybrid_new.log

# python examples/qualcomm/oss_scripts/qwen3/test_model.py -m SM8650 -b build-android -c --model_dir /workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4 --mode both --layer_idx 0 --prefill_ar_len 32 --max_seq_len 32 -v -a ./qwen3_attention_only_qnn 2>&1 | tee qwen3_attention_ab.log
# python examples/qualcomm/oss_scripts/qwen3/test_model.py -m SM8650 -b build-android -c --model_dir /workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4 --layer_idx 0 --prefill_ar_len 32 --max_seq_len 32 --ptq 16a4w --focus_matmul -a ./qwen3_attention_only_qnn 2>&1 | tee qwen3_attention_only_layer0_focus_matmul.log
# python test_model.py -m SM8650 -b build-android -c -a ./qwen3_one_layer_qnn --prefill_ar_len 32 --max_seq_len 32 --hidden_size 512 --intermediate_size 1024 --num_attention_heads 8 --num_key_value_heads 2

# adb shell "cd ${DEVICE_DIR} \
#  && export LD_LIBRARY_PATH=${DEVICE_DIR} \
#  && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
#  && ./qnn_executor_runner --model_path ./kv_llama_qnn.pte"

export PTE_DIR=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4/qwen_qnn
python examples/qualcomm/oss_scripts/qwen3/qwen3.py \
 -b build-android \
 -s a50cde57 \
 -m SM8650 \
 --ptq 16a4w \
 --model_dir ${MODEL_DIR} \
 --tokenizer_model ${MODEL_DIR}/tokenizer.json \
 --model_mode hybrid \
 --max_seq_len 128 \
 --prompt "Are you conscious?" \
 --llama_model qwen3 \
 --artifact ${OUT_DIR} \
 --use_tman \
 --num_sharding 1 \
 --kv_updater shift_pointer \
 --pre_gen_pte ${PTE_DIR} \
 --device "a50cde57"


export PTE_DIR=/workspaces/tman/models/Qwen3-1.7B-GPTQ-Int4/qwen_qnn_new
python examples/qualcomm/oss_scripts/qwen3/qwen3_new.py \
 -b build-android \
 -s a50cde57 \
 -m SM8650 \
 --ptq 16a4w \
 --model_dir ${MODEL_DIR} \
 --tokenizer_model ${MODEL_DIR}/tokenizer.json \
 --model_mode hybrid \
 --max_seq_len 128 \
 --prompt "Are you conscious?" \
 --llama_model qwen3 \
 --artifact ${OUT_DIR} \
 --use_tman \
 --num_sharding 1 \
 --kv_updater shift_pointer \
 --pre_gen_pte ${PTE_DIR} \
 --device "a50cde57"

adb kill-server
adb -a nodaemon server start