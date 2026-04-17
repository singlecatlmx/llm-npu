//==============================================================================
// TMAN W4A8 FP16 pipeline correctness test
//
// Target pipeline:
//   TMANPrecomputeA8 -> TMANLinearA8 -> TMANFinalize
//
// Operator-level scope in op_gen:
// - x       : FP16,  [1, 1, 1, K]
// - l       : UINT8, [1, 1, 1, packed_bytes]
// - qweight : INT32, [1, 1, P_tiles, ints_per_tile]
// - scales  : INT32, [1, 1, P_tiles, ints_per_tile]
// - c       : UINT8, [1, 1, 1, M * bits * sizeof(float)]
// - y       : FP16,  [1, 1, 1, M]
//
// This matches the engineering README goal:
// - fp16 in / fp16 out
// - W4G128 symmetric
// - activation quantized to per-tensor A8 inside TMANPrecomputeA8
//
// Modes:
// 1. x86 HTP simulator
// 2. x86 offline context compilation
// 3. device execution from context binary
//==============================================================================

#include <dlfcn.h>

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnTensor.h"
#include "QnnTypes.h"

namespace {

using fp16_t = uint16_t;

constexpr const char* kPackageName = "TMANOpPackage";
constexpr const char* kProviderName = "TMANOpPackageInterfaceProvider";

constexpr int32_t kBits = 4;
constexpr int32_t kGroupSize = 128;
constexpr int32_t kSymmetric = 1;

constexpr int32_t kPackGroup = 4;
constexpr int32_t kTileQ = 64;
constexpr int32_t kVecP = 128;
constexpr int32_t kVecC = 32;
constexpr int32_t kVecQ = 4;

struct QnnApi {
  QnnBackend_CreateFn_t backendCreate;
  QnnBackend_FreeFn_t backendFree;
  QnnBackend_RegisterOpPackageFn_t registerOpPackage;
  QnnContext_CreateFn_t contextCreate;
  QnnContext_FreeFn_t contextFree;
  QnnContext_GetBinarySizeFn_t contextGetBinarySize;
  QnnContext_GetBinaryFn_t contextGetBinary;
  QnnContext_CreateFromBinaryFn_t contextCreateFromBinary;
  QnnGraph_CreateFn_t graphCreate;
  QnnGraph_AddNodeFn_t graphAddNode;
  QnnGraph_FinalizeFn_t graphFinalize;
  QnnGraph_ExecuteFn_t graphExecute;
  QnnGraph_RetrieveFn_t graphRetrieve;
  QnnTensor_CreateGraphTensorFn_t tensorCreateGraphTensor;
  QnnDevice_CreateFn_t deviceCreate;
  QnnDevice_FreeFn_t deviceFree;
  QnnLog_CreateFn_t logCreate;
  QnnLog_FreeFn_t logFree;
};

struct ContextGuard {
  const QnnApi* api = nullptr;
  Qnn_ContextHandle_t ctx = nullptr;

  ~ContextGuard() {
    if (api != nullptr && ctx != nullptr) {
      api->contextFree(ctx, nullptr);
    }
  }
};

struct CaseSpec {
  std::string name;
  std::string label;
  int32_t rows = 1;
  int32_t gemm_m = 0;
  int32_t gemm_k = 0;
  int32_t tile_p = 0;
  float tolerance = 0.0f;
  // input_scale_factor 用来在测试里人为偏移动态 input_scale，
  // 以验证 runtime 是否真的使用了外部 input_scale，而不是偷偷按输入重算。
  float input_scale_factor = 1.0f;
  // output_scale 用来驱动 FinalizeA8 的量化反量化逻辑。
  float output_scale = 1.0f;
  bool zero_input = false;
  bool zero_weight = false;
  bool wide_dynamic = false;

  std::vector<fp16_t> x;
  std::vector<uint8_t> weight_codes;
  std::vector<fp16_t> group_scales;
};

struct PreparedCase {
  int32_t num_groups = 0;
  int32_t qweight_tiles = 0;
  int32_t qweight_int32_per_tile = 0;
  int32_t scales_int32_per_tile = 0;
  int32_t l_size_bytes = 0;
  int32_t c_size_bytes = 0;
  float input_scale = 1.0f;
  float output_scale = 1.0f;

  std::vector<int32_t> qweight_repacked;
  std::vector<int32_t> scales_repacked;
  std::vector<fp16_t> expected_y;
};

struct GraphRecord {
  Qnn_GraphHandle_t graph = nullptr;
  Qnn_Tensor_t x = QNN_TENSOR_INIT;
  Qnn_Tensor_t y = QNN_TENSOR_INIT;
  uint32_t x_dims[4] = {0, 0, 0, 0};
  uint32_t y_dims[4] = {0, 0, 0, 0};
  std::string x_name;
  std::string y_name;
};

struct TestSummary {
  int passed = 0;
  int failed = 0;
};

static std::map<std::string, GraphRecord> g_graph_records;

static fp16_t f32_to_f16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
  const int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) {
    return sign;
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

static float f16_to_f32(fp16_t h) {
  const uint32_t sign = (static_cast<uint32_t>((h >> 15) & 1u)) << 31;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t u = 0;
  if (exp == 0u) {
    u = (mant == 0u) ? sign : (sign | ((1u + 112u) << 23) | (mant << 13));
  } else if (exp == 31u) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

static int8_t round_clamp_to_int8(float value) {
  value = std::round(value);
  if (value > 127.0f) value = 127.0f;
  if (value < -127.0f) value = -127.0f;
  return static_cast<int8_t>(value);
}

static bool check_qnn(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::fprintf(
        stderr,
        "[FATAL] %s failed: error=0x%lx\n",
        what,
        static_cast<unsigned long>(err));
    return false;
  }
  return true;
}

static bool register_op_package_or_report(
    const QnnApi& api,
    Qnn_BackendHandle_t backend,
    const char* pkg_lib,
    const char* target) {
  const Qnn_ErrorHandle_t err =
      api.registerOpPackage(backend, pkg_lib, kProviderName, target);
  if (err == QNN_SUCCESS) {
    return true;
  }

  std::fprintf(
      stderr,
      "[FATAL] registerOpPackage failed: error=0x%lx\n"
      "        package: %s\n"
      "        provider: %s\n"
      "        target: %s\n"
      "        stage: op package loading, before graph creation\n",
      static_cast<unsigned long>(err),
      pkg_lib,
      kProviderName,
      (target == nullptr) ? "<default>" : target);
  return false;
}

static void qnn_log_callback(
    const char* fmt,
    QnnLog_Level_t level,
    uint64_t /*ts*/,
    va_list args) {
  if (level > QNN_LOG_LEVEL_WARN) {
    return;
  }
  const char* level_str = "?";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      level_str = "E";
      break;
    case QNN_LOG_LEVEL_WARN:
      level_str = "W";
      break;
    case QNN_LOG_LEVEL_INFO:
      level_str = "I";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      level_str = "D";
      break;
    default:
      break;
  }
  std::printf("[QNN/%s] ", level_str);
  std::vprintf(fmt, args);
  std::printf("\n");
}

static bool load_qnn_api(const char* htp_lib_path, void** handle_out, QnnApi* api) {
  void* handle = dlopen(htp_lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    std::fprintf(
        stderr,
        "[FATAL] Cannot load HTP backend: %s\n  dlerror: %s\n",
        htp_lib_path,
        dlerror());
    return false;
  }
  *handle_out = handle;

  using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto getProviders =
      reinterpret_cast<GetProvidersFn>(dlsym(handle, "QnnInterface_getProviders"));
  if (getProviders == nullptr) {
    std::fprintf(stderr, "[FATAL] QnnInterface_getProviders not found\n");
    return false;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t num_providers = 0;
  const Qnn_ErrorHandle_t err = getProviders(&providers, &num_providers);
  if (err != QNN_SUCCESS || providers == nullptr || num_providers == 0 || providers[0] == nullptr) {
    std::fprintf(
        stderr,
        "[FATAL] QnnInterface_getProviders failed: 0x%lx\n",
        static_cast<unsigned long>(err));
    return false;
  }

  auto& impl = providers[0]->QNN_INTERFACE_VER_NAME;
  api->backendCreate = impl.backendCreate;
  api->backendFree = impl.backendFree;
  api->registerOpPackage = impl.backendRegisterOpPackage;
  api->contextCreate = impl.contextCreate;
  api->contextFree = impl.contextFree;
  api->contextGetBinarySize = impl.contextGetBinarySize;
  api->contextGetBinary = impl.contextGetBinary;
  api->contextCreateFromBinary = impl.contextCreateFromBinary;
  api->graphCreate = impl.graphCreate;
  api->graphAddNode = impl.graphAddNode;
  api->graphFinalize = impl.graphFinalize;
  api->graphExecute = impl.graphExecute;
  api->graphRetrieve = impl.graphRetrieve;
  api->tensorCreateGraphTensor = impl.tensorCreateGraphTensor;
  api->deviceCreate = impl.deviceCreate;
  api->deviceFree = impl.deviceFree;
  api->logCreate = impl.logCreate;
  api->logFree = impl.logFree;
  return true;
}

static int32_t get_l_size_bytes(int32_t k, int32_t group_size) {
  const int32_t l_size = (k / kPackGroup) * 16;
  const int32_t ls_size = k / 256;
  const int32_t lb_size = k / group_size;
  return l_size * static_cast<int32_t>(sizeof(int16_t)) +
      std::max(ls_size * static_cast<int32_t>(sizeof(float)), 128) +
      std::max(lb_size * static_cast<int32_t>(sizeof(float)), 128);
}

static int32_t get_c_size_bytes(int32_t m, int32_t bits) {
  return m * bits * static_cast<int32_t>(sizeof(float));
}

static Qnn_Param_t make_scalar_param(const char* name, int32_t value) {
  Qnn_Param_t param = QNN_PARAM_INIT;
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.name = name;
  param.scalarParam.dataType = QNN_DATATYPE_INT_32;
  param.scalarParam.int32Value = value;
  return param;
}

static Qnn_Tensor_t make_tensor(
    const char* name,
    Qnn_TensorType_t type,
    Qnn_DataType_t data_type,
    uint32_t* dims,
    uint32_t rank,
    const void* static_data,
    uint32_t static_bytes) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_1;
  tensor.v1.id = 0;
  tensor.v1.name = name;
  tensor.v1.type = type;
  tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v1.dataType = data_type;
  tensor.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  tensor.v1.rank = rank;
  tensor.v1.dimensions = dims;
  tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
  if (type == QNN_TENSOR_TYPE_STATIC) {
    tensor.v1.clientBuf = {
        const_cast<void*>(static_data),
        static_cast<uint32_t>(static_bytes)};
  } else {
    tensor.v1.clientBuf = {nullptr, 0};
  }
  return tensor;
}

static CaseSpec make_case(
    const char* name,
    const char* label,
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t tile_p,
    float tolerance,
    float input_scale_factor = 1.0f,
    float output_scale = 1.0f,
    bool zero_input = false,
    bool zero_weight = false,
    bool wide_dynamic = false) {
  CaseSpec spec;
  spec.name = name;
  spec.label = label;
  spec.rows = 1;
  spec.gemm_m = gemm_m;
  spec.gemm_k = gemm_k;
  spec.tile_p = tile_p;
  spec.tolerance = tolerance;
  spec.input_scale_factor = input_scale_factor;
  spec.output_scale = output_scale;
  spec.zero_input = zero_input;
  spec.zero_weight = zero_weight;
  spec.wide_dynamic = wide_dynamic;

  spec.x.resize(static_cast<size_t>(spec.rows) * gemm_k);
  spec.weight_codes.resize(static_cast<size_t>(gemm_m) * gemm_k);
  spec.group_scales.resize(static_cast<size_t>(gemm_m) * (gemm_k / kGroupSize));

  for (int32_t k = 0; k < gemm_k; ++k) {
    float x_value = 0.0f;
    if (!zero_input) {
      const int32_t pattern = ((k * 13 + 5) % 17) - 8;
      const float scale = wide_dynamic ? ((k % 5 == 0) ? 1.75f : 0.15f) : 0.25f;
      x_value = static_cast<float>(pattern) * scale;
    }
    spec.x[k] = f32_to_f16(x_value);
  }

  for (int32_t out = 0; out < gemm_m; ++out) {
    for (int32_t k = 0; k < gemm_k; ++k) {
      uint8_t code = 8u;
      if (!zero_weight) {
        code = static_cast<uint8_t>((out * 11 + k * 7 + 3) % 16);
      }
      spec.weight_codes[static_cast<size_t>(out) * gemm_k + k] = code;
    }
  }

  const int32_t num_groups = gemm_k / kGroupSize;
  for (int32_t out = 0; out < gemm_m; ++out) {
    for (int32_t group = 0; group < num_groups; ++group) {
      const float scale = 0.03125f * static_cast<float>(1 + ((out + group) % 3));
      spec.group_scales[static_cast<size_t>(out) * num_groups + group] =
          f32_to_f16(scale);
    }
  }

  return spec;
}

static bool pack_old_fp16_qweight(
    const CaseSpec& spec,
    int32_t bits,
    int32_t tile_p,
    std::vector<int32_t>* packed,
    int32_t* num_tiles,
    int32_t* ints_per_tile) {
  if (packed == nullptr || num_tiles == nullptr || ints_per_tile == nullptr) {
    return false;
  }
  if (bits != 4 || tile_p <= 0 || (spec.gemm_k % kPackGroup) != 0 ||
      (spec.gemm_k / kPackGroup) % kTileQ != 0 || (spec.gemm_m % kVecP) != 0) {
    return false;
  }

  const int32_t q = spec.gemm_k / kPackGroup;
  const int32_t p = spec.gemm_m * bits;
  if ((p % tile_p) != 0 || (tile_p % kVecP) != 0) {
    return false;
  }

  const int32_t p_blocks = p / kVecP;
  const int32_t p_tiles = p / tile_p;
  const int32_t q_tiles = q / kTileQ;
  const int32_t tile_p_blocks = tile_p / kVecP;
  const int32_t tile_q_blocks = kTileQ / kVecQ;

  std::vector<uint8_t> idx(static_cast<size_t>(spec.gemm_m) * bits * q, 0);
  for (int32_t out = 0; out < spec.gemm_m; ++out) {
    for (int32_t bit = 0; bit < bits; ++bit) {
      for (int32_t q_idx = 0; q_idx < q; ++q_idx) {
        uint8_t nibble = 0;
        const int32_t k_base = q_idx * kPackGroup;
        for (int32_t lane = 0; lane < kPackGroup; ++lane) {
          const uint8_t code =
              spec.weight_codes[static_cast<size_t>(out) * spec.gemm_k + k_base + lane];
          nibble |= static_cast<uint8_t>(((code >> bit) & 0x1u) << lane);
        }
        idx[(static_cast<size_t>(out) * bits + bit) * q + q_idx] = nibble;
      }
    }
  }

  std::vector<uint8_t> tmp1(static_cast<size_t>(p) * q, 0);
  const int32_t m_blocks = spec.gemm_m / kVecP;
  for (int32_t m_block = 0; m_block < m_blocks; ++m_block) {
    for (int32_t bit = 0; bit < bits; ++bit) {
      const int32_t p_block = m_block * bits + bit;
      for (int32_t m_local = 0; m_local < kVecP; ++m_local) {
        const int32_t out = m_block * kVecP + m_local;
        for (int32_t q_idx = 0; q_idx < q; ++q_idx) {
          tmp1[(static_cast<size_t>(p_block) * kVecP + m_local) * q + q_idx] =
              idx[(static_cast<size_t>(out) * bits + bit) * q + q_idx];
        }
      }
    }
  }

  std::vector<uint8_t> interleaved(static_cast<size_t>(p) * q, 0);
  for (int32_t p_block = 0; p_block < p_blocks; ++p_block) {
    for (int32_t a = 0; a < 2; ++a) {
      for (int32_t b = 0; b < 2; ++b) {
        for (int32_t c = 0; c < kVecC; ++c) {
          const int32_t src_i = ((a * 2 + b) * kVecC) + c;
          const int32_t dst_i = ((b * kVecC + c) * 2) + a;
          for (int32_t q_idx = 0; q_idx < q; ++q_idx) {
            interleaved[(static_cast<size_t>(p_block) * kVecP + dst_i) * q + q_idx] =
                tmp1[(static_cast<size_t>(p_block) * kVecP + src_i) * q + q_idx];
          }
        }
      }
    }
  }

  std::vector<uint8_t> bytes;
  bytes.reserve(
      static_cast<size_t>(p_tiles) * q_tiles * tile_p_blocks * tile_q_blocks *
      (kVecQ / 2) * kVecP);
  for (int32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
    for (int32_t q_tile = 0; q_tile < q_tiles; ++q_tile) {
      for (int32_t p_vec = 0; p_vec < tile_p_blocks; ++p_vec) {
        for (int32_t q_vec = 0; q_vec < tile_q_blocks; ++q_vec) {
          for (int32_t pair = 0; pair < (kVecQ / 2); ++pair) {
            for (int32_t vp = 0; vp < kVecP; ++vp) {
              const int32_t p_idx = p_tile * tile_p + p_vec * kVecP + vp;
              const int32_t q0 = q_tile * kTileQ + q_vec * kVecQ + pair * 2 + 0;
              const int32_t q1 = q_tile * kTileQ + q_vec * kVecQ + pair * 2 + 1;
              const uint8_t low = interleaved[static_cast<size_t>(p_idx) * q + q0];
              const uint8_t high = interleaved[static_cast<size_t>(p_idx) * q + q1];
              bytes.push_back(static_cast<uint8_t>(low | (high << 4)));
            }
          }
        }
      }
    }
  }

  if ((bytes.size() % sizeof(int32_t)) != 0) {
    return false;
  }

  packed->assign(bytes.size() / sizeof(int32_t), 0);
  std::memcpy(packed->data(), bytes.data(), bytes.size());
  *num_tiles = p_tiles;
  *ints_per_tile = static_cast<int32_t>(packed->size() / p_tiles);
  return true;
}

static bool pack_old_fp16_scales(
    const CaseSpec& spec,
    int32_t tile_p,
    std::vector<int32_t>* packed,
    int32_t* ints_per_tile) {
  if (packed == nullptr || ints_per_tile == nullptr) {
    return false;
  }

  const int32_t num_groups = spec.gemm_k / kGroupSize;
  const int32_t q = spec.gemm_k / kPackGroup;
  const int32_t p = spec.gemm_m * kBits;
  const int32_t p_tiles = p / tile_p;
  const int32_t q_tiles = q / kTileQ;
  const int32_t q_group_size = kGroupSize / kPackGroup;
  const int32_t tile_p_m = tile_p / kBits;
  const int32_t tile_m_blocks = tile_p_m / kVecP;
  const int32_t q_groups_per_tile = kTileQ / q_group_size;

  if ((spec.gemm_k % kGroupSize) != 0 || (q % kTileQ) != 0 ||
      (tile_p % (kBits * kVecP)) != 0) {
    return false;
  }

  std::vector<fp16_t> fp16_values;
  fp16_values.reserve(
      static_cast<size_t>(p_tiles) * q_tiles * tile_m_blocks * q_groups_per_tile * kVecP);

  for (int32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
    for (int32_t q_tile = 0; q_tile < q_tiles; ++q_tile) {
      for (int32_t m_block = 0; m_block < tile_m_blocks; ++m_block) {
        for (int32_t q_group = 0; q_group < q_groups_per_tile; ++q_group) {
          for (int32_t vp = 0; vp < kVecP; ++vp) {
            const int32_t out = p_tile * tile_p_m + m_block * kVecP + vp;
            const int32_t group = q_tile * q_groups_per_tile + q_group;
            if (out < 0 || out >= spec.gemm_m || group < 0 || group >= num_groups) {
              return false;
            }
            fp16_values.push_back(
                spec.group_scales[static_cast<size_t>(out) * num_groups + group]);
          }
        }
      }
    }
  }

  const size_t fp16_bytes = fp16_values.size() * sizeof(fp16_t);
  if ((fp16_bytes % sizeof(int32_t)) != 0) {
    return false;
  }

  packed->assign(fp16_bytes / sizeof(int32_t), 0);
  std::memcpy(packed->data(), fp16_values.data(), fp16_bytes);
  *ints_per_tile = static_cast<int32_t>(packed->size() / p_tiles);
  return true;
}

static PreparedCase prepare_case(const CaseSpec& spec) {
  PreparedCase prepared;
  prepared.num_groups = spec.gemm_k / kGroupSize;
  prepared.l_size_bytes = get_l_size_bytes(spec.gemm_k, kGroupSize);
  prepared.c_size_bytes = get_c_size_bytes(spec.gemm_m, kBits);
  prepared.output_scale = spec.output_scale;

  const bool qweight_ok = pack_old_fp16_qweight(
      spec,
      kBits,
      spec.tile_p,
      &prepared.qweight_repacked,
      &prepared.qweight_tiles,
      &prepared.qweight_int32_per_tile);
  const bool scales_ok = pack_old_fp16_scales(
      spec,
      spec.tile_p,
      &prepared.scales_repacked,
      &prepared.scales_int32_per_tile);
  if (!qweight_ok || !scales_ok) {
    std::fprintf(stderr, "[FATAL] Failed to repack W4A8 weights/scales\n");
    std::abort();
  }

  prepared.expected_y.assign(
      static_cast<size_t>(spec.rows) * spec.gemm_m,
      f32_to_f16(0.0f));

  // 这里的 golden 刻意与正式 runtime 的数值路径保持一致：
  // 1. 先按外部 input_scale 做输入量化；
  // 2. 用反量化后的输入和 GPTQ 权重做浮点 reference；
  // 3. 最后再按 output_scale 做一轮输出量化反量化。
  std::vector<int8_t> q_a(spec.gemm_k, 0);
  float amax = 0.0f;
  for (int32_t k = 0; k < spec.gemm_k; ++k) {
    amax = std::max(amax, std::fabs(f16_to_f32(spec.x[k])));
  }
  const float s_a_dynamic = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
  const float s_a = std::max(s_a_dynamic * spec.input_scale_factor, 1.0f / 127.0f);
  prepared.input_scale = s_a;
  const float inv_s_a = (s_a == 0.0f) ? 0.0f : (1.0f / s_a);
  for (int32_t k = 0; k < spec.gemm_k; ++k) {
    q_a[k] = round_clamp_to_int8(f16_to_f32(spec.x[k]) * inv_s_a);
  }

  for (int32_t out = 0; out < spec.gemm_m; ++out) {
    float acc = 0.0f;
    for (int32_t group = 0; group < prepared.num_groups; ++group) {
      const float group_scale = f16_to_f32(
          spec.group_scales[static_cast<size_t>(out) * prepared.num_groups + group]);
      const int32_t k_begin = group * kGroupSize;
      const int32_t k_end = k_begin + kGroupSize;
      for (int32_t k = k_begin; k < k_end; ++k) {
        const int32_t w =
            static_cast<int32_t>(spec.weight_codes[static_cast<size_t>(out) * spec.gemm_k + k]) - 8;
        acc += static_cast<float>(q_a[k]) * s_a * static_cast<float>(w) * group_scale;
      }
    }
    // runtime FinalizeA8 是“bit-serial -> fp16 -> output quant/dequant”。
    // 因此 golden 这里也先显式经过一次 fp16 舍入，避免边界 case 上
    // 因为比较对象不一致而出现假失败。
    const float acc_fp16 = f16_to_f32(f32_to_f16(acc));
    const int8_t y_q = round_clamp_to_int8(acc_fp16 / prepared.output_scale);
    prepared.expected_y[out] = f32_to_f16(static_cast<float>(y_q) * prepared.output_scale);
  }

  return prepared;
}

static bool compare_fp16_output(
    const CaseSpec& spec,
    const PreparedCase& prepared,
    const std::vector<fp16_t>& actual) {
  for (size_t i = 0; i < actual.size(); ++i) {
    const float value = f16_to_f32(actual[i]);
    const float expected = f16_to_f32(prepared.expected_y[i]);
    const float diff = std::fabs(value - expected);
    if (diff > spec.tolerance) {
      const int32_t row = static_cast<int32_t>(i / spec.gemm_m);
      const int32_t col = static_cast<int32_t>(i % spec.gemm_m);
      std::printf(
          "    mismatch at row=%d col=%d: expected=%.8f actual=%.8f diff=%.8f\n",
          row,
          col,
          expected,
          value,
          diff);
      return false;
    }
  }
  return true;
}

static std::vector<CaseSpec> get_cases() {
  return {
      make_case(
          "w4a8_fp16_basic",
          "TMAN W4A8 FP16 全链路: 基线 K256/M128",
          128,
          256,
          512,
          0.20f),
      make_case(
          "w4a8_fp16_zero_input",
          "TMAN W4A8 FP16 全链路: zero-input",
          128,
          256,
          512,
          1e-3f,
          1.0f,
          1.0f,
          true,
          false,
          false),
      make_case(
          "w4a8_fp16_zero_weight",
          "TMAN W4A8 FP16 全链路: zero-weight",
          128,
          256,
          512,
          1e-3f,
          1.0f,
          1.0f,
          false,
          true,
          false),
      make_case(
          "w4a8_fp16_multigroup",
          "TMAN W4A8 FP16 全链路: 多 group K512/M128 + 外部 input_scale",
          128,
          512,
          512,
          0.30f,
          0.75f,
          1.0f,
          false,
          false,
          true),
      make_case(
          "w4a8_fp16_output_scale",
          "TMAN W4A8 FP16 全链路: 外部 output_scale 量化反量化",
          128,
          256,
          512,
          0.251f,
          1.0f,
          0.25f,
          false,
          false,
          false),
      make_case(
          "w4a8_fp16_autosplit",
          "TMAN W4A8 FP16 全链路: autosplit 大输出通道 M256",
          256,
          256,
          1024,
          0.20f),
  };
}

static bool build_pipeline_graph(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const CaseSpec& spec,
    const PreparedCase& prepared) {
  const std::string graph_name = spec.name + "_pipeline_graph";
  Qnn_GraphHandle_t graph = nullptr;
  if (!check_qnn(api.graphCreate(ctx, graph_name.c_str(), nullptr, &graph), "graphCreate")) {
    return false;
  }

  GraphRecord& rec = g_graph_records[graph_name];
  rec.graph = graph;
  rec.x_name = spec.name + "_x";
  rec.y_name = spec.name + "_y";
  rec.x_dims[0] = 1u;
  rec.x_dims[1] = 1u;
  rec.x_dims[2] = static_cast<uint32_t>(spec.rows);
  rec.x_dims[3] = static_cast<uint32_t>(spec.gemm_k);
  rec.y_dims[0] = 1u;
  rec.y_dims[1] = 1u;
  rec.y_dims[2] = static_cast<uint32_t>(spec.rows);
  rec.y_dims[3] = static_cast<uint32_t>(spec.gemm_m);

  uint32_t l_dims[4] = {
      1u,
      1u,
      1u,
      static_cast<uint32_t>(prepared.l_size_bytes)};
  uint32_t qweight_dims[4] = {
      1u,
      1u,
      static_cast<uint32_t>(prepared.qweight_tiles),
      static_cast<uint32_t>(prepared.qweight_int32_per_tile)};
  uint32_t scales_dims[4] = {
      1u,
      1u,
      static_cast<uint32_t>(prepared.qweight_tiles),
      static_cast<uint32_t>(prepared.scales_int32_per_tile)};
  uint32_t act_scale_dims[4] = {1u, 1u, 1u, 1u};
  uint32_t c_dims[4] = {
      1u,
      1u,
      1u,
      static_cast<uint32_t>(prepared.c_size_bytes)};

  const std::string l_name = spec.name + "_l";
  const std::string qweight_name = spec.name + "_qweight";
  const std::string scales_name = spec.name + "_scales";
  const std::string input_scale_name = spec.name + "_input_scale";
  const std::string output_scale_name = spec.name + "_output_scale";
  const std::string c_name = spec.name + "_c";

  rec.x = make_tensor(
      rec.x_name.c_str(),
      QNN_TENSOR_TYPE_APP_WRITE,
      QNN_DATATYPE_FLOAT_16,
      rec.x_dims,
      4,
      nullptr,
      0);
  Qnn_Tensor_t l_tensor = make_tensor(
      l_name.c_str(),
      QNN_TENSOR_TYPE_NATIVE,
      QNN_DATATYPE_UINT_8,
      l_dims,
      4,
      nullptr,
      0);
  Qnn_Tensor_t qweight_tensor = make_tensor(
      qweight_name.c_str(),
      QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_INT_32,
      qweight_dims,
      4,
      prepared.qweight_repacked.data(),
      static_cast<uint32_t>(prepared.qweight_repacked.size() * sizeof(int32_t)));
  Qnn_Tensor_t scales_tensor = make_tensor(
      scales_name.c_str(),
      QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_INT_32,
      scales_dims,
      4,
      prepared.scales_repacked.data(),
      static_cast<uint32_t>(prepared.scales_repacked.size() * sizeof(int32_t)));
  Qnn_Tensor_t input_scale_tensor = make_tensor(
      input_scale_name.c_str(),
      QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_FLOAT_32,
      act_scale_dims,
      4,
      &prepared.input_scale,
      static_cast<uint32_t>(sizeof(float)));
  Qnn_Tensor_t output_scale_tensor = make_tensor(
      output_scale_name.c_str(),
      QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_FLOAT_32,
      act_scale_dims,
      4,
      &prepared.output_scale,
      static_cast<uint32_t>(sizeof(float)));
  Qnn_Tensor_t c_tensor = make_tensor(
      c_name.c_str(),
      QNN_TENSOR_TYPE_NATIVE,
      QNN_DATATYPE_UINT_8,
      c_dims,
      4,
      nullptr,
      0);
  rec.y = make_tensor(
      rec.y_name.c_str(),
      QNN_TENSOR_TYPE_APP_READ,
      QNN_DATATYPE_FLOAT_16,
      rec.y_dims,
      4,
      nullptr,
      0);

  if (!check_qnn(api.tensorCreateGraphTensor(graph, &rec.x), "createTensor(x)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &l_tensor), "createTensor(l)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &qweight_tensor), "createTensor(qweight)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &scales_tensor), "createTensor(scales)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &input_scale_tensor), "createTensor(input_scale)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &output_scale_tensor), "createTensor(output_scale)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &c_tensor), "createTensor(c)")) return false;
  if (!check_qnn(api.tensorCreateGraphTensor(graph, &rec.y), "createTensor(y)")) return false;

  Qnn_Param_t params[3] = {
      make_scalar_param("group_size", kGroupSize),
      make_scalar_param("bits", kBits),
      make_scalar_param("symmetric", kSymmetric),
  };

  Qnn_Tensor_t precompute_inputs[] = {rec.x, input_scale_tensor};
  Qnn_Tensor_t precompute_outputs[] = {l_tensor};
  Qnn_OpConfig_t precompute = QNN_OPCONFIG_INIT;
  precompute.v1.name = "tman_precompute_a8_fp16";
  precompute.v1.packageName = kPackageName;
  precompute.v1.typeName = "TMANPrecomputeA8";
  precompute.v1.numOfParams = 3;
  precompute.v1.params = params;
  precompute.v1.numOfInputs = 2;
  precompute.v1.inputTensors = precompute_inputs;
  precompute.v1.numOfOutputs = 1;
  precompute.v1.outputTensors = precompute_outputs;
  if (!check_qnn(api.graphAddNode(graph, precompute), "graphAddNode(TMANPrecomputeA8)")) {
    return false;
  }

  Qnn_Tensor_t linear_inputs[] = {l_tensor, qweight_tensor, scales_tensor};
  Qnn_Tensor_t linear_outputs[] = {c_tensor};
  Qnn_OpConfig_t linear = QNN_OPCONFIG_INIT;
  linear.v1.name = "tman_linear_a8_fp16";
  linear.v1.packageName = kPackageName;
  linear.v1.typeName = "TMANLinearA8";
  linear.v1.numOfParams = 3;
  linear.v1.params = params;
  linear.v1.numOfInputs = 3;
  linear.v1.inputTensors = linear_inputs;
  linear.v1.numOfOutputs = 1;
  linear.v1.outputTensors = linear_outputs;
  if (!check_qnn(api.graphAddNode(graph, linear), "graphAddNode(TMANLinearA8)")) {
    return false;
  }

  Qnn_Tensor_t finalize_inputs[] = {c_tensor, output_scale_tensor};
  // FinalizeA8 是这次正式链路新增的“轻量 output-dequant 变体”。
  // 它让层外继续保持 fp16，同时把 output_scale 真正纳入数值路径。
  Qnn_Tensor_t finalize_outputs[] = {rec.y};
  Qnn_OpConfig_t finalize = QNN_OPCONFIG_INIT;
  finalize.v1.name = "tman_finalize_fp16";
  finalize.v1.packageName = kPackageName;
  finalize.v1.typeName = "TMANFinalizeA8";
  finalize.v1.numOfParams = 3;
  finalize.v1.params = params;
  finalize.v1.numOfInputs = 2;
  finalize.v1.inputTensors = finalize_inputs;
  finalize.v1.numOfOutputs = 1;
  finalize.v1.outputTensors = finalize_outputs;
  if (!check_qnn(api.graphAddNode(graph, finalize), "graphAddNode(TMANFinalizeA8)")) {
    return false;
  }

  if (!check_qnn(api.graphFinalize(graph, nullptr, nullptr), "graphFinalize")) {
    return false;
  }

  return true;
}

static bool run_pipeline_case(
    const QnnApi& api,
    Qnn_BackendHandle_t backend,
    const CaseSpec& spec) {
  const PreparedCase prepared = prepare_case(spec);
  ContextGuard context{&api, nullptr};
  if (!check_qnn(api.contextCreate(backend, nullptr, nullptr, &context.ctx), "contextCreate")) {
    return false;
  }

  g_graph_records.clear();
  if (!build_pipeline_graph(api, context.ctx, spec, prepared)) {
    return false;
  }

  const std::string graph_name = spec.name + "_pipeline_graph";
  auto it = g_graph_records.find(graph_name);
  if (it == g_graph_records.end()) {
    return false;
  }

  std::vector<fp16_t> actual(static_cast<size_t>(spec.rows) * spec.gemm_m, 0);
  Qnn_Tensor_t exec_input = it->second.x;
  exec_input.v1.clientBuf = {
      const_cast<fp16_t*>(spec.x.data()),
      static_cast<uint32_t>(spec.x.size() * sizeof(fp16_t))};
  Qnn_Tensor_t exec_output = it->second.y;
  exec_output.v1.clientBuf = {
      actual.data(),
      static_cast<uint32_t>(actual.size() * sizeof(fp16_t))};

  if (!check_qnn(
          api.graphExecute(
              it->second.graph,
              &exec_input,
              1,
              &exec_output,
              1,
              nullptr,
              nullptr),
          "graphExecute(TMANA8 fp16 pipeline)")) {
    return false;
  }

  return compare_fp16_output(spec, prepared, actual);
}

static bool save_graph_metadata(const char* bin_path) {
  std::string meta_path(bin_path);
  const auto dot = meta_path.rfind('.');
  if (dot != std::string::npos) {
    meta_path = meta_path.substr(0, dot) + ".meta";
  } else {
    meta_path += ".meta";
  }

  FILE* f = std::fopen(meta_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "[FATAL] Cannot write metadata: %s\n", meta_path.c_str());
    return false;
  }

  for (const auto& [graph_name, rec] : g_graph_records) {
    std::fprintf(
        f,
        "%s %u %s %u %s %u %u %u %u %u %u %u %u\n",
        graph_name.c_str(),
        rec.x.v1.id,
        rec.x.v1.name,
        rec.y.v1.id,
        rec.y.v1.name,
        rec.x_dims[0],
        rec.x_dims[1],
        rec.x_dims[2],
        rec.x_dims[3],
        rec.y_dims[0],
        rec.y_dims[1],
        rec.y_dims[2],
        rec.y_dims[3]);
  }

  std::fclose(f);
  std::printf("[INFO] Metadata saved: %s (%zu graphs)\n", meta_path.c_str(), g_graph_records.size());
  return true;
}

static bool load_graph_metadata_and_retrieve(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const char* bin_path) {
  std::string meta_path(bin_path);
  const auto dot = meta_path.rfind('.');
  if (dot != std::string::npos) {
    meta_path = meta_path.substr(0, dot) + ".meta";
  } else {
    meta_path += ".meta";
  }

  FILE* f = std::fopen(meta_path.c_str(), "r");
  if (f == nullptr) {
    std::fprintf(stderr, "[FATAL] Cannot open metadata: %s\n", meta_path.c_str());
    return false;
  }

  char line[512];
  int count = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
      continue;
    }

    char gname[128];
    char xname[128];
    char yname[128];
    uint32_t x_id = 0;
    uint32_t y_id = 0;
    uint32_t xd0 = 0, xd1 = 0, xd2 = 0, xd3 = 0;
    uint32_t yd0 = 0, yd1 = 0, yd2 = 0, yd3 = 0;
    if (std::sscanf(
            line,
            "%127s %u %127s %u %127s %u %u %u %u %u %u %u %u",
            gname,
            &x_id,
            xname,
            &y_id,
            yname,
            &xd0,
            &xd1,
            &xd2,
            &xd3,
            &yd0,
            &yd1,
            &yd2,
            &yd3) != 13) {
      continue;
    }

    Qnn_GraphHandle_t graph = nullptr;
    const Qnn_ErrorHandle_t err = api.graphRetrieve(ctx, gname, &graph);
    if (err != QNN_SUCCESS) {
      std::fprintf(
          stderr,
          "[WARN] graphRetrieve(%s) failed: 0x%lx\n",
          gname,
          static_cast<unsigned long>(err));
      continue;
    }

    GraphRecord& rec = g_graph_records[gname];
    rec.graph = graph;
    rec.x_name = xname;
    rec.y_name = yname;
    rec.x_dims[0] = xd0;
    rec.x_dims[1] = xd1;
    rec.x_dims[2] = xd2;
    rec.x_dims[3] = xd3;
    rec.y_dims[0] = yd0;
    rec.y_dims[1] = yd1;
    rec.y_dims[2] = yd2;
    rec.y_dims[3] = yd3;

    rec.x = make_tensor(
        rec.x_name.c_str(),
        QNN_TENSOR_TYPE_APP_WRITE,
        QNN_DATATYPE_FLOAT_16,
        rec.x_dims,
        4,
        nullptr,
        0);
    rec.x.v1.id = x_id;
    rec.y = make_tensor(
        rec.y_name.c_str(),
        QNN_TENSOR_TYPE_APP_READ,
        QNN_DATATYPE_FLOAT_16,
        rec.y_dims,
        4,
        nullptr,
        0);
    rec.y.v1.id = y_id;

    ++count;
  }

  std::fclose(f);
  std::printf("[INFO] Loaded %d graphs from metadata\n", count);
  return count > 0;
}

static bool execute_pipeline_graph(
    const QnnApi& api,
    const std::string& graph_name,
    const fp16_t* x_data,
    uint32_t x_bytes,
    fp16_t* y_data,
    uint32_t y_bytes) {
  auto it = g_graph_records.find(graph_name);
  if (it == g_graph_records.end()) {
    std::fprintf(stderr, "[ERROR] Graph '%s' not found in graph record map\n", graph_name.c_str());
    return false;
  }

  Qnn_Tensor_t x = it->second.x;
  x.v1.clientBuf = {const_cast<fp16_t*>(x_data), x_bytes};

  Qnn_Tensor_t y = it->second.y;
  y.v1.clientBuf = {y_data, y_bytes};

  return api.graphExecute(it->second.graph, &x, 1, &y, 1, nullptr, nullptr) == QNN_SUCCESS;
}

static TestSummary run_pipeline_correctness_tests(
    const QnnApi& api,
    const std::vector<CaseSpec>& cases) {
  TestSummary summary;
  std::printf("========================================\n");
  std::printf("  TMAN W4A8 FP16 全链路测试\n");
  std::printf("========================================\n");

  for (const auto& spec : cases) {
    const PreparedCase prepared = prepare_case(spec);
    const std::string graph_name = spec.name + "_pipeline_graph";
    std::vector<fp16_t> actual(static_cast<size_t>(spec.rows) * spec.gemm_m, 0);

    std::printf("\n[TEST] %s\n", spec.label.c_str());
    std::printf(
        "       shape: rows=%d out=%d in=%d tile_p=%d p_tiles=%d\n",
        spec.rows,
        spec.gemm_m,
        spec.gemm_k,
        spec.tile_p,
        prepared.qweight_tiles);

    bool ok = execute_pipeline_graph(
        api,
        graph_name,
        spec.x.data(),
        static_cast<uint32_t>(spec.x.size() * sizeof(fp16_t)),
        actual.data(),
        static_cast<uint32_t>(actual.size() * sizeof(fp16_t)));
    if (ok) {
      ok = compare_fp16_output(spec, prepared, actual);
    } else {
      std::fprintf(stderr, "[FATAL] graphExecute failed for graph: %s\n", graph_name.c_str());
    }

    std::printf("       %s\n", ok ? "[PASS]" : "[FAIL]");
    if (ok) {
      ++summary.passed;
    } else {
      ++summary.failed;
    }
  }

  std::printf("\n========================================\n");
  std::printf("  Passed: %d\n", summary.passed);
  std::printf("  Failed: %d\n", summary.failed);
  std::printf(
      "  Result: %s\n",
      summary.failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  std::printf("========================================\n");
  return summary;
}

static int mode_compile(const char* htp_lib, const char* pkg_lib, const char* output_bin) {
  std::printf("========================================\n");
  std::printf("  TMAN W4A8 FP16 离线编译\n");
  std::printf("========================================\n\n");

  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api)) {
    return 1;
  }

  Qnn_LogHandle_t log_handle = nullptr;
  api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle);

  Qnn_BackendHandle_t backend = nullptr;
  if (!check_qnn(api.backendCreate(log_handle, nullptr, &backend), "backendCreate")) {
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  if (!register_op_package_or_report(api, backend, pkg_lib, nullptr)) {
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  Qnn_ContextHandle_t ctx = nullptr;
  if (!check_qnn(api.contextCreate(backend, nullptr, nullptr, &ctx), "contextCreate")) {
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  g_graph_records.clear();
  const auto cases = get_cases();
  for (const auto& spec : cases) {
    const PreparedCase prepared = prepare_case(spec);
    if (!build_pipeline_graph(api, ctx, spec, prepared)) {
      api.contextFree(ctx, nullptr);
      api.backendFree(backend);
      if (log_handle != nullptr) api.logFree(log_handle);
      dlclose(htp_handle);
      return 1;
    }
  }

  Qnn_ContextBinarySize_t bin_size = 0;
  if (!check_qnn(api.contextGetBinarySize(ctx, &bin_size), "contextGetBinarySize")) {
    api.contextFree(ctx, nullptr);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  std::vector<uint8_t> buffer(bin_size);
  Qnn_ContextBinarySize_t written = 0;
  if (!check_qnn(api.contextGetBinary(ctx, buffer.data(), bin_size, &written), "contextGetBinary")) {
    api.contextFree(ctx, nullptr);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  FILE* f = std::fopen(output_bin, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[FATAL] Cannot open %s for writing\n", output_bin);
    api.contextFree(ctx, nullptr);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }
  std::fwrite(buffer.data(), 1, static_cast<size_t>(written), f);
  std::fclose(f);
  std::printf("[INFO] Context binary saved: %s (%lu bytes)\n", output_bin, (unsigned long)written);

  if (!save_graph_metadata(output_bin)) {
    api.contextFree(ctx, nullptr);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  api.contextFree(ctx, nullptr);
  api.backendFree(backend);
  if (log_handle != nullptr) api.logFree(log_handle);
  dlclose(htp_handle);
  return 0;
}

static int mode_execute(const char* htp_lib, const char* pkg_lib, const char* context_bin) {
  std::printf("========================================\n");
  std::printf("  TMAN W4A8 FP16 设备执行\n");
  std::printf("========================================\n\n");

  FILE* f = std::fopen(context_bin, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "[FATAL] Cannot open context binary: %s\n", context_bin);
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  const long fsize = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> bin_data(static_cast<size_t>(fsize));
  if (std::fread(bin_data.data(), 1, static_cast<size_t>(fsize), f) != static_cast<size_t>(fsize)) {
    std::fprintf(stderr, "[FATAL] Failed to read context binary: %s\n", context_bin);
    std::fclose(f);
    return 1;
  }
  std::fclose(f);

  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api)) {
    return 1;
  }

  Qnn_LogHandle_t log_handle = nullptr;
  api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle);

  Qnn_BackendHandle_t backend = nullptr;
  if (!check_qnn(api.backendCreate(log_handle, nullptr, &backend), "backendCreate")) {
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  QnnHtpDevice_CustomConfig_t htp_dev_config{};
  htp_dev_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
  htp_dev_config.arch.deviceId = 0;
  htp_dev_config.arch.arch = QNN_HTP_DEVICE_ARCH_V75;

  QnnDevice_Config_t dev_config{};
  dev_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
  dev_config.customConfig = &htp_dev_config;
  const QnnDevice_Config_t* dev_configs[] = {&dev_config, nullptr};

  Qnn_DeviceHandle_t device = nullptr;
  if (!check_qnn(api.deviceCreate(log_handle, dev_configs, &device), "deviceCreate")) {
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  if (!register_op_package_or_report(api, backend, pkg_lib, "HTP")) {
    api.deviceFree(device);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  Qnn_ContextHandle_t ctx = nullptr;
  if (!check_qnn(
          api.contextCreateFromBinary(
              backend,
              device,
              nullptr,
              bin_data.data(),
              static_cast<Qnn_ContextBinarySize_t>(bin_data.size()),
              &ctx,
              nullptr),
          "contextCreateFromBinary")) {
    api.deviceFree(device);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  g_graph_records.clear();
  if (!load_graph_metadata_and_retrieve(api, ctx, context_bin)) {
    api.contextFree(ctx, nullptr);
    api.deviceFree(device);
    api.backendFree(backend);
    if (log_handle != nullptr) api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  const TestSummary summary = run_pipeline_correctness_tests(api, get_cases());

  api.contextFree(ctx, nullptr);
  api.deviceFree(device);
  api.backendFree(backend);
  if (log_handle != nullptr) api.logFree(log_handle);
  dlclose(htp_handle);
  return summary.failed == 0 ? 0 : 1;
}

static int mode_simulator(const char* htp_lib, const char* pkg_lib) {
  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api)) {
    return 1;
  }

  Qnn_LogHandle_t log_handle = nullptr;
  if (!check_qnn(api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle), "logCreate")) {
    dlclose(htp_handle);
    return 1;
  }

  Qnn_BackendHandle_t backend = nullptr;
  if (!check_qnn(api.backendCreate(log_handle, nullptr, &backend), "backendCreate")) {
    api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  if (!register_op_package_or_report(api, backend, pkg_lib, nullptr)) {
    api.backendFree(backend);
    api.logFree(log_handle);
    dlclose(htp_handle);
    return 1;
  }

  TestSummary summary;
  std::printf("========================================\n");
  std::printf("  TMAN W4A8 FP16 全链路测试\n");
  std::printf("========================================\n");
  for (const auto& spec : get_cases()) {
    const PreparedCase prepared = prepare_case(spec);
    std::printf("\n[TEST] %s\n", spec.label.c_str());
    std::printf(
        "       shape: rows=%d out=%d in=%d tile_p=%d p_tiles=%d\n",
        spec.rows,
        spec.gemm_m,
        spec.gemm_k,
        spec.tile_p,
        prepared.qweight_tiles);
    const bool ok = run_pipeline_case(api, backend, spec);
    std::printf("       %s\n", ok ? "[PASS]" : "[FAIL]");
    if (ok) {
      ++summary.passed;
    } else {
      ++summary.failed;
    }
  }

  std::printf("\n========================================\n");
  std::printf("  Passed: %d\n", summary.passed);
  std::printf("  Failed: %d\n", summary.failed);
  std::printf("  Result: %s\n", summary.failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  std::printf("========================================\n");

  api.backendFree(backend);
  api.logFree(log_handle);
  dlclose(htp_handle);
  return summary.failed == 0 ? 0 : 1;
}

}  // namespace

static void usage(const char* prog) {
  std::printf("TMAN W4A8 FP16 全链路测试\n\n");
  std::printf("Usage:\n");
  std::printf("  x86 模拟器模式:\n");
  std::printf("    %s <libQnnHtp.so> <op_pkg.so>\n\n", prog);
  std::printf("  离线编译 context:\n");
  std::printf("    %s --compile <libQnnHtp.so> <x86_op_pkg.so> <output.bin>\n\n", prog);
  std::printf("  设备执行:\n");
  std::printf("    %s --execute <libQnnHtp.so> <hex_op_pkg.so> <context.bin>\n", prog);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argc > 0 ? argv[0] : "test_tman_linear_a8");
    return 1;
  }

  const std::string arg1 = argv[1];
  if (arg1 == "--compile") {
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    }
    return mode_compile(argv[2], argv[3], argv[4]);
  }

  if (arg1 == "--execute") {
    if (argc < 5) {
      usage(argv[0]);
      return 1;
    }
    return mode_execute(argv[2], argv[3], argv[4]);
  }

  return mode_simulator(argv[1], argv[2]);
}
