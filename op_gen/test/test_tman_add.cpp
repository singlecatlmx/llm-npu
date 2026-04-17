//==============================================================================
// TMANAdd Unit Test & Performance Benchmark
//
// 三种运行模式:
//   1. 模拟器模式 (x86 一体化): 编译+执行一步完成
//      ./test_tman_add <libQnnHtp.so> <op_pkg.so>
//
//   2. 离线编译模式 (x86 host): 构建图、finalize → 保存 context binary
//      ./test_tman_add --compile <libQnnHtp.so> <x86_op_pkg.so> <output.bin>
//
//   3. 设备执行模式 (aarch64 on-device): 加载 context binary → execute
//      ./test_tman_add --execute <libQnnHtp.so> <hex_op_pkg.so> <context.bin>
//
// QNN HTP 架构说明:
//   在设备端, graphAddNode 需要 ARM 侧 prepare 引擎的 OpValidatorRegistry
//   支持, 但 registerOpPackage(target="HTP") 只向 DSP 侧发送 .so 路径,
//   不注册 ARM 侧的 OpValidator。因此设备端必须使用离线编译的 context binary.
//   这也是 ExecuTorch 等框架采用的部署方式。
//==============================================================================

#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// QNN SDK headers
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnProfile.h"
#include "QnnTensor.h"
#include "QnnTypes.h"

//==============================================================================
// Portable fp16 <-> fp32 (不依赖 __fp16)
//==============================================================================
using fp16_t = uint16_t;

static fp16_t f32_to_f16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
  int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0)
    return sign;
  if (exp >= 31)
    return sign | 0x7C00u;
  return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static float f16_to_f32(fp16_t h) {
  uint32_t sign = (uint32_t)((h >> 15) & 1u) << 31;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t u;
  if (exp == 0u) {
    u = (mant == 0u) ? sign : (sign | ((1u + 112u) << 23) | (mant << 13));
  } else if (exp == 31u) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float r;
  memcpy(&r, &u, sizeof(r));
  return r;
}

//==============================================================================
// QNN API 封装
//==============================================================================

#define QNN_CHECK(expr, msg)                  \
  do {                                        \
    Qnn_ErrorHandle_t _err = (expr);          \
    if (_err != QNN_SUCCESS) {                \
      fprintf(                                \
          stderr,                             \
          "[FATAL] %s failed: error=0x%lx\n", \
          msg,                                \
          (unsigned long)_err);               \
      return false;                           \
    }                                         \
  } while (0)

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
  QnnProfile_CreateFn_t profileCreate;
  QnnProfile_FreeFn_t profileFree;
  QnnProfile_GetEventsFn_t profileGetEvents;
  QnnProfile_GetSubEventsFn_t profileGetSubEvents;
  QnnProfile_GetExtendedEventDataFn_t profileGetExtendedEventData;
};

static void qnn_log_callback(
    const char* fmt,
    QnnLog_Level_t level,
    uint64_t /*ts*/,
    va_list args) {
  if (level <= QNN_LOG_LEVEL_WARN) {
    const char* lvl_str = "?";
    switch (level) {
      case QNN_LOG_LEVEL_ERROR:
        lvl_str = "E";
        break;
      case QNN_LOG_LEVEL_WARN:
        lvl_str = "W";
        break;
      case QNN_LOG_LEVEL_INFO:
        lvl_str = "I";
        break;
      case QNN_LOG_LEVEL_DEBUG:
        lvl_str = "D";
        break;
      default:
        break;
    }
    printf("[QNN/%s] ", lvl_str);
    vprintf(fmt, args);
    printf("\n");
  }
}

static double now_us() {
  using namespace std::chrono;
  return (double)duration_cast<microseconds>(
             steady_clock::now().time_since_epoch())
      .count();
}

//==============================================================================
// 图定义 — 所有测试用例和 benchmark 共享
//==============================================================================

struct GraphDef {
  const char* name;
  uint32_t batch, height, width, channels;
  bool has_old_graph; // benchmark 需要旧方案对比图
  const char* old_name; // 旧方案图名
};

// 存储每个图的 tensor 信息 (tensorCreateGraphTensor 会分配 ID)
struct GraphTensors {
  Qnn_GraphHandle_t graph;
  Qnn_Tensor_t t_a;
  Qnn_Tensor_t t_b;
  Qnn_Tensor_t t_out;
  uint32_t dims[4]; // 保持 dims 生命周期
  std::string name_a, name_b, name_out; // tensor 名称持久存储
};

#include <map>
static std::map<std::string, GraphTensors> g_graph_tensors;

static const GraphDef g_test_graphs[] = {
    {"test_basic", 1, 1, 1, 128, false, nullptr},
    {"test_zero", 1, 1, 1, 64, false, nullptr},
    {"test_cancel", 1, 1, 1, 64, false, nullptr},
    {"test_large", 1, 1, 1, 4096, false, nullptr},
    {"test_nonaligned", 1, 1, 1, 100, false, nullptr},
};
static const int g_num_test_graphs = 5;

static const GraphDef g_bench_graphs[] = {
    {"bench_4096_1", 1, 1, 1, 4096, true, "bench_old_4096_1"},
    {"bench_4096_64", 1, 1, 64, 4096, true, "bench_old_4096_64"},
};
static const int g_num_bench_graphs = 2;

// Pipeline benchmark: K 个算子串联在一个图中
// 中间结果以 NATIVE tensor 留在 TCM (TMANAdd) 或 MainMemory (ElemAdd)
// 消除了 isolated benchmark 中人为的 DMA(clientBuf↔TCM) 开销
// 模拟真实 LLM pipeline: 每个 residual add 的 a 来自上一个 op 的输出
struct ChainBenchDef {
  const char* name;
  const char* old_name;
  uint32_t batch, height, width, channels;
  int chain_len;
};

static const ChainBenchDef g_chain_bench[] = {
    {"bench_chain_4096_1", "bench_chain_old_4096_1", 1, 1, 1, 4096, 20},
    {"bench_chain_4096_64", "bench_chain_old_4096_64", 1, 1, 64, 4096, 2},
};
static const int g_num_chain_bench = 2;

//==============================================================================
// 元数据持久化: 编译时保存 tensor IDs, 设备执行时加载
//==============================================================================
static bool save_graph_metadata(const char* bin_path) {
  std::string meta_path(bin_path);
  auto dot = meta_path.rfind('.');
  if (dot != std::string::npos)
    meta_path = meta_path.substr(0, dot) + ".meta";
  else
    meta_path += ".meta";

  FILE* f = fopen(meta_path.c_str(), "w");
  if (!f) {
    fprintf(stderr, "[FATAL] Cannot write metadata: %s\n", meta_path.c_str());
    return false;
  }

  for (auto& [name, gt] : g_graph_tensors) {
    fprintf(
        f,
        "%s %u %s %u %s %u %s %u %u %u %u\n",
        name.c_str(),
        gt.t_a.v1.id,
        gt.t_a.v1.name,
        gt.t_b.v1.id,
        gt.t_b.v1.name,
        gt.t_out.v1.id,
        gt.t_out.v1.name,
        gt.dims[0],
        gt.dims[1],
        gt.dims[2],
        gt.dims[3]);
  }

  fclose(f);
  printf(
      "[INFO] Metadata saved: %s (%zu graphs)\n",
      meta_path.c_str(),
      g_graph_tensors.size());
  return true;
}

static bool load_graph_metadata_and_retrieve(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const char* bin_path) {
  std::string meta_path(bin_path);
  auto dot = meta_path.rfind('.');
  if (dot != std::string::npos)
    meta_path = meta_path.substr(0, dot) + ".meta";
  else
    meta_path += ".meta";

  FILE* f = fopen(meta_path.c_str(), "r");
  if (!f) {
    fprintf(stderr, "[FATAL] Cannot open metadata: %s\n", meta_path.c_str());
    return false;
  }

  char line[512];
  int count = 0;
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
      continue;

    char gname[128], aname[128], bname[128], oname[128];
    uint32_t a_id, b_id, o_id, batch, height, width, channels;

    if (sscanf(
            line,
            "%127s %u %127s %u %127s %u %127s %u %u %u %u",
            gname,
            &a_id,
            aname,
            &b_id,
            bname,
            &o_id,
            oname,
            &batch,
            &height,
            &width,
            &channels) != 11) {
      continue;
    }

    // 从 context binary 恢复图句柄
    Qnn_GraphHandle_t graph = nullptr;
    Qnn_ErrorHandle_t err = api.graphRetrieve(ctx, gname, &graph);
    if (err != QNN_SUCCESS) {
      fprintf(
          stderr,
          "[WARN] graphRetrieve(%s) failed: 0x%lx\n",
          gname,
          (unsigned long)err);
      continue;
    }

    // 重建 GraphTensors, 使用编译时分配的 tensor IDs
    GraphTensors& gt = g_graph_tensors[gname];
    gt.graph = graph;
    gt.dims[0] = batch;
    gt.dims[1] = height;
    gt.dims[2] = width;
    gt.dims[3] = channels;

    gt.name_a = aname;
    gt.name_b = bname;
    gt.name_out = oname;

    gt.t_a = QNN_TENSOR_INIT;
    gt.t_a.version = QNN_TENSOR_VERSION_1;
    gt.t_a.v1.id = a_id;
    gt.t_a.v1.name = gt.name_a.c_str();
    gt.t_a.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    gt.t_a.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    gt.t_a.v1.dataType = QNN_DATATYPE_FLOAT_16;
    gt.t_a.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    gt.t_a.v1.rank = 4;
    gt.t_a.v1.dimensions = gt.dims;
    gt.t_a.v1.memType = QNN_TENSORMEMTYPE_RAW;
    gt.t_a.v1.clientBuf = {nullptr, 0};

    gt.t_b = QNN_TENSOR_INIT;
    gt.t_b.version = QNN_TENSOR_VERSION_1;
    gt.t_b.v1.id = b_id;
    gt.t_b.v1.name = gt.name_b.c_str();
    gt.t_b.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    gt.t_b.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    gt.t_b.v1.dataType = QNN_DATATYPE_FLOAT_16;
    gt.t_b.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    gt.t_b.v1.rank = 4;
    gt.t_b.v1.dimensions = gt.dims;
    gt.t_b.v1.memType = QNN_TENSORMEMTYPE_RAW;
    gt.t_b.v1.clientBuf = {nullptr, 0};

    gt.t_out = QNN_TENSOR_INIT;
    gt.t_out.version = QNN_TENSOR_VERSION_1;
    gt.t_out.v1.id = o_id;
    gt.t_out.v1.name = gt.name_out.c_str();
    gt.t_out.v1.type = QNN_TENSOR_TYPE_APP_READ;
    gt.t_out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    gt.t_out.v1.dataType = QNN_DATATYPE_FLOAT_16;
    gt.t_out.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    gt.t_out.v1.rank = 4;
    gt.t_out.v1.dimensions = gt.dims;
    gt.t_out.v1.memType = QNN_TENSORMEMTYPE_RAW;
    gt.t_out.v1.clientBuf = {nullptr, 0};

    count++;
    printf(
        "[INFO] Retrieved graph: %s (IDs: %s=%u %s=%u %s=%u)\n",
        gname,
        aname,
        a_id,
        bname,
        b_id,
        oname,
        o_id);
  }

  fclose(f);
  printf("[INFO] Loaded %d graphs from metadata\n", count);
  return count > 0;
}

//==============================================================================
// 构建图 (编译模式 + 模拟器模式使用)
//==============================================================================
static bool build_tman_graph(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const GraphDef& gd) {
  Qnn_GraphHandle_t graph = nullptr;
  QNN_CHECK(api.graphCreate(ctx, gd.name, nullptr, &graph), "graphCreate");

  GraphTensors& gt = g_graph_tensors[gd.name];
  gt.graph = graph;
  gt.dims[0] = gd.batch;
  gt.dims[1] = gd.height;
  gt.dims[2] = gd.width;
  gt.dims[3] = gd.channels;

  gt.t_a = QNN_TENSOR_INIT;
  gt.t_a.version = QNN_TENSOR_VERSION_1;
  gt.t_a.v1.id = 0;
  gt.t_a.v1.name = "a";
  gt.t_a.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_a.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_a.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_a.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_a.v1.rank = 4;
  gt.t_a.v1.dimensions = gt.dims;
  gt.t_a.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_a.v1.clientBuf = {nullptr, 0};

  gt.t_b = QNN_TENSOR_INIT;
  gt.t_b.version = QNN_TENSOR_VERSION_1;
  gt.t_b.v1.id = 0;
  gt.t_b.v1.name = "b";
  gt.t_b.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_b.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_b.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_b.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_b.v1.rank = 4;
  gt.t_b.v1.dimensions = gt.dims;
  gt.t_b.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_b.v1.clientBuf = {nullptr, 0};

  gt.t_out = QNN_TENSOR_INIT;
  gt.t_out.version = QNN_TENSOR_VERSION_1;
  gt.t_out.v1.id = 0;
  gt.t_out.v1.name = "out";
  gt.t_out.v1.type = QNN_TENSOR_TYPE_APP_READ;
  gt.t_out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_out.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_out.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_out.v1.rank = 4;
  gt.t_out.v1.dimensions = gt.dims;
  gt.t_out.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_out.v1.clientBuf = {nullptr, 0};

  // tensorCreateGraphTensor 会分配 ID 到 tensor 结构中
  QNN_CHECK(api.tensorCreateGraphTensor(graph, &gt.t_a), "createTensor(a)");
  QNN_CHECK(api.tensorCreateGraphTensor(graph, &gt.t_b), "createTensor(b)");
  QNN_CHECK(api.tensorCreateGraphTensor(graph, &gt.t_out), "createTensor(out)");

  Qnn_Tensor_t inputs[] = {gt.t_a, gt.t_b};
  Qnn_Tensor_t outputs[] = {gt.t_out};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "tman_add_0";
  op.v1.packageName = "TMANOpPackage";
  op.v1.typeName = "TMANAdd";
  op.v1.numOfParams = 0;
  op.v1.params = nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = outputs;

  QNN_CHECK(api.graphAddNode(graph, op), "graphAddNode(TMANAdd)");
  QNN_CHECK(api.graphFinalize(graph, nullptr, nullptr), "graphFinalize");

  printf(
      "[INFO] Built graph: %s [%u,%u,%u,%u] (IDs: a=%u b=%u out=%u)\n",
      gd.name,
      gd.batch,
      gd.height,
      gd.width,
      gd.channels,
      gt.t_a.v1.id,
      gt.t_b.v1.id,
      gt.t_out.v1.id);
  return true;
}

static bool build_old_graph(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const GraphDef& gd) {
  if (!gd.has_old_graph || !gd.old_name)
    return true;

  Qnn_GraphHandle_t graph = nullptr;
  QNN_CHECK(
      api.graphCreate(ctx, gd.old_name, nullptr, &graph), "graphCreate(old)");

  GraphTensors& gt = g_graph_tensors[gd.old_name];
  gt.graph = graph;
  gt.dims[0] = gd.batch;
  gt.dims[1] = gd.height;
  gt.dims[2] = gd.width;
  gt.dims[3] = gd.channels;

  gt.t_a = QNN_TENSOR_INIT;
  gt.t_a.version = QNN_TENSOR_VERSION_1;
  gt.t_a.v1.id = 0;
  gt.t_a.v1.name = "old_a";
  gt.t_a.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_a.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_a.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_a.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_a.v1.rank = 4;
  gt.t_a.v1.dimensions = gt.dims;
  gt.t_a.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_a.v1.clientBuf = {nullptr, 0};

  gt.t_b = QNN_TENSOR_INIT;
  gt.t_b.version = QNN_TENSOR_VERSION_1;
  gt.t_b.v1.id = 0;
  gt.t_b.v1.name = "old_b";
  gt.t_b.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_b.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_b.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_b.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_b.v1.rank = 4;
  gt.t_b.v1.dimensions = gt.dims;
  gt.t_b.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_b.v1.clientBuf = {nullptr, 0};

  gt.t_out = QNN_TENSOR_INIT;
  gt.t_out.version = QNN_TENSOR_VERSION_1;
  gt.t_out.v1.id = 0;
  gt.t_out.v1.name = "old_out";
  gt.t_out.v1.type = QNN_TENSOR_TYPE_APP_READ;
  gt.t_out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_out.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_out.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_out.v1.rank = 4;
  gt.t_out.v1.dimensions = gt.dims;
  gt.t_out.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_out.v1.clientBuf = {nullptr, 0};

  QNN_CHECK(api.tensorCreateGraphTensor(graph, &gt.t_a), "createTensor(old_a)");
  QNN_CHECK(api.tensorCreateGraphTensor(graph, &gt.t_b), "createTensor(old_b)");
  QNN_CHECK(
      api.tensorCreateGraphTensor(graph, &gt.t_out), "createTensor(old_out)");

  Qnn_Tensor_t inputs[] = {gt.t_a, gt.t_b};
  Qnn_Tensor_t outputs[] = {gt.t_out};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "builtin_add_0";
  op.v1.packageName = "qti.aisw";
  op.v1.typeName = "ElementWiseAdd";
  op.v1.numOfParams = 0;
  op.v1.params = nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = outputs;

  QNN_CHECK(api.graphAddNode(graph, op), "graphAddNode(old)");
  QNN_CHECK(api.graphFinalize(graph, nullptr, nullptr), "graphFinalize(old)");

  printf(
      "[INFO] Built old graph: %s (IDs: a=%u b=%u out=%u)\n",
      gd.old_name,
      gt.t_a.v1.id,
      gt.t_b.v1.id,
      gt.t_out.v1.id);
  return true;
}

//==============================================================================
// 构建链式图: K 个 op 串联, 中间 NATIVE tensor 留在框架管理的内存
//
// TMANAdd chain: a(TCM)→op0→t0(TCM)→op1→t1(TCM)→...→out(TCM)
//   只有 a 需要 DMA(clientBuf→TCM), 中间全在 TCM, 零额外 DMA
//
// ElemAdd chain: a(Main)→op0→t0(Main)→op1→t1(Main)→...→out(Main)
//   全在 MainMemory, 无 DMA 开销 (与 standalone ElemAdd 相同)
//
// 这使得对比公平: 都是 1× FastRPC + K× pure kernel
//==============================================================================
static bool build_chained_graph(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx,
    const char* graph_name,
    const char* pkg_name,
    const char* type_name,
    const char* a_name,
    const char* b_name,
    const char* out_name,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    int chain_len) {
  Qnn_GraphHandle_t graph = nullptr;
  QNN_CHECK(api.graphCreate(ctx, graph_name, nullptr, &graph), "graphCreate");

  GraphTensors& gt = g_graph_tensors[graph_name];
  gt.graph = graph;
  gt.dims[0] = batch;
  gt.dims[1] = height;
  gt.dims[2] = width;
  gt.dims[3] = channels;
  gt.name_a = a_name;
  gt.name_b = b_name;
  gt.name_out = out_name;

  // Input a (APP_WRITE)
  gt.t_a = QNN_TENSOR_INIT;
  gt.t_a.version = QNN_TENSOR_VERSION_1;
  gt.t_a.v1.id = 0;
  gt.t_a.v1.name = gt.name_a.c_str();
  gt.t_a.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_a.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_a.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_a.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_a.v1.rank = 4;
  gt.t_a.v1.dimensions = gt.dims;
  gt.t_a.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_a.v1.clientBuf = {nullptr, 0};

  // Input b (APP_WRITE, shared by all ops)
  gt.t_b = QNN_TENSOR_INIT;
  gt.t_b.version = QNN_TENSOR_VERSION_1;
  gt.t_b.v1.id = 0;
  gt.t_b.v1.name = gt.name_b.c_str();
  gt.t_b.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  gt.t_b.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  gt.t_b.v1.dataType = QNN_DATATYPE_FLOAT_16;
  gt.t_b.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  gt.t_b.v1.rank = 4;
  gt.t_b.v1.dimensions = gt.dims;
  gt.t_b.v1.memType = QNN_TENSORMEMTYPE_RAW;
  gt.t_b.v1.clientBuf = {nullptr, 0};

  QNN_CHECK(
      api.tensorCreateGraphTensor(graph, &gt.t_a), "createTensor(chain_a)");
  QNN_CHECK(
      api.tensorCreateGraphTensor(graph, &gt.t_b), "createTensor(chain_b)");

  // Persist names for intermediate tensors and ops
  std::vector<std::string> t_names(chain_len);
  std::vector<std::string> op_names(chain_len);

  Qnn_Tensor_t prev = gt.t_a;

  for (int k = 0; k < chain_len; k++) {
    bool is_last = (k == chain_len - 1);
    Qnn_Tensor_t cur_out;

    if (is_last) {
      // Final output (APP_READ → DMA back to clientBuf)
      gt.t_out = QNN_TENSOR_INIT;
      gt.t_out.version = QNN_TENSOR_VERSION_1;
      gt.t_out.v1.id = 0;
      gt.t_out.v1.name = gt.name_out.c_str();
      gt.t_out.v1.type = QNN_TENSOR_TYPE_APP_READ;
      gt.t_out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      gt.t_out.v1.dataType = QNN_DATATYPE_FLOAT_16;
      gt.t_out.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
      gt.t_out.v1.rank = 4;
      gt.t_out.v1.dimensions = gt.dims;
      gt.t_out.v1.memType = QNN_TENSORMEMTYPE_RAW;
      gt.t_out.v1.clientBuf = {nullptr, 0};
      QNN_CHECK(
          api.tensorCreateGraphTensor(graph, &gt.t_out),
          "createTensor(chain_out)");
      cur_out = gt.t_out;
    } else {
      // Intermediate NATIVE tensor (stays in framework-managed memory)
      // For TMANAdd: constrained to TCM (no DMA between ops!)
      // For ElemAdd: in MainMemory
      char buf[32];
      snprintf(buf, sizeof(buf), "t%d", k);
      t_names[k] = buf;

      cur_out = QNN_TENSOR_INIT;
      cur_out.version = QNN_TENSOR_VERSION_1;
      cur_out.v1.id = 0;
      cur_out.v1.name = t_names[k].c_str();
      cur_out.v1.type = QNN_TENSOR_TYPE_NATIVE;
      cur_out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      cur_out.v1.dataType = QNN_DATATYPE_FLOAT_16;
      cur_out.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
      cur_out.v1.rank = 4;
      cur_out.v1.dimensions = gt.dims;
      cur_out.v1.memType = QNN_TENSORMEMTYPE_RAW;
      cur_out.v1.clientBuf = {nullptr, 0};
      QNN_CHECK(
          api.tensorCreateGraphTensor(graph, &cur_out),
          "createTensor(chain_inter)");
    }

    // op(prev, b) → cur_out
    Qnn_Tensor_t inputs[] = {prev, gt.t_b};
    Qnn_Tensor_t outputs[] = {cur_out};

    char op_buf[64];
    snprintf(op_buf, sizeof(op_buf), "%s_%d", type_name, k);
    op_names[k] = op_buf;

    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = op_names[k].c_str();
    op.v1.packageName = pkg_name;
    op.v1.typeName = type_name;
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = outputs;

    QNN_CHECK(api.graphAddNode(graph, op), "graphAddNode(chain)");
    prev = cur_out;
  }

  QNN_CHECK(api.graphFinalize(graph, nullptr, nullptr), "graphFinalize(chain)");

  printf(
      "[INFO] Built chained graph: %s [%u,%u,%u,%u] x%d ops "
      "(IDs: a=%u b=%u out=%u)\n",
      graph_name,
      batch,
      height,
      width,
      channels,
      chain_len,
      gt.t_a.v1.id,
      gt.t_b.v1.id,
      gt.t_out.v1.id);
  return true;
}

//==============================================================================
// 通用: 加载 QNN API
//==============================================================================
static bool
load_qnn_api(const char* htp_lib_path, void** htp_handle_out, QnnApi* api) {
  void* handle = dlopen(htp_lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    fprintf(
        stderr,
        "[FATAL] Cannot load HTP backend: %s\n  dlerror: %s\n",
        htp_lib_path,
        dlerror());
    return false;
  }
  *htp_handle_out = handle;
  printf("[INFO] Loaded HTP backend: %s\n", htp_lib_path);

  using GetProvidersFn =
      Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto getProviders =
      (GetProvidersFn)dlsym(handle, "QnnInterface_getProviders");
  if (!getProviders) {
    fprintf(stderr, "[FATAL] QnnInterface_getProviders not found\n");
    return false;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  Qnn_ErrorHandle_t err = getProviders(&providers, &numProviders);
  if (err != QNN_SUCCESS || numProviders < 1 || !providers || !providers[0]) {
    fprintf(stderr, "[FATAL] getProviders failed: 0x%lx\n", (unsigned long)err);
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
  api->profileCreate = impl.profileCreate;
  api->profileFree = impl.profileFree;
  api->profileGetEvents = impl.profileGetEvents;
  api->profileGetSubEvents = impl.profileGetSubEvents;
  api->profileGetExtendedEventData = impl.profileGetExtendedEventData;
  return true;
}

//==============================================================================
// 模式 1: 离线编译 (--compile)
//==============================================================================
static int
mode_compile(const char* htp_lib, const char* pkg_lib, const char* output_bin) {
  printf("========================================\n");
  printf("  TMANAdd Offline Compilation\n");
  printf("========================================\n\n");

  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api))
    return 1;

  Qnn_LogHandle_t log_handle = nullptr;
  api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle);

  Qnn_BackendHandle_t backend = nullptr;
  if (api.backendCreate(log_handle, nullptr, &backend) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] backendCreate failed\n");
    return 1;
  }

  if (api.registerOpPackage(
          backend, pkg_lib, "TMANOpPackageInterfaceProvider", nullptr) !=
      QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] registerOpPackage failed\n");
    return 1;
  }
  printf("[INFO] Op package registered: %s\n", pkg_lib);

  Qnn_ContextHandle_t ctx = nullptr;
  if (api.contextCreate(backend, nullptr, nullptr, &ctx) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] contextCreate failed\n");
    return 1;
  }

  printf("\n[INFO] Building test graphs...\n");
  for (int i = 0; i < g_num_test_graphs; i++) {
    if (!build_tman_graph(api, ctx, g_test_graphs[i])) {
      fprintf(
          stderr, "[FATAL] Failed to build graph: %s\n", g_test_graphs[i].name);
      return 1;
    }
  }

  printf("\n[INFO] Building benchmark graphs...\n");
  for (int i = 0; i < g_num_bench_graphs; i++) {
    if (!build_tman_graph(api, ctx, g_bench_graphs[i])) {
      fprintf(
          stderr,
          "[FATAL] Failed to build bench graph: %s\n",
          g_bench_graphs[i].name);
      return 1;
    }
    if (!build_old_graph(api, ctx, g_bench_graphs[i])) {
      fprintf(
          stderr,
          "[FATAL] Failed to build old graph: %s\n",
          g_bench_graphs[i].old_name);
      return 1;
    }
  }

  printf("\n[INFO] Building pipeline (chained) benchmark graphs...\n");
  for (int i = 0; i < g_num_chain_bench; i++) {
    const auto& cb = g_chain_bench[i];
    if (!build_chained_graph(
            api,
            ctx,
            cb.name,
            "TMANOpPackage",
            "TMANAdd",
            "ca",
            "cb",
            "cout",
            cb.batch,
            cb.height,
            cb.width,
            cb.channels,
            cb.chain_len)) {
      fprintf(stderr, "[FATAL] Failed to build chain graph: %s\n", cb.name);
      return 1;
    }
    if (!build_chained_graph(
            api,
            ctx,
            cb.old_name,
            "qti.aisw",
            "ElementWiseAdd",
            "old_ca",
            "old_cb",
            "old_cout",
            cb.batch,
            cb.height,
            cb.width,
            cb.channels,
            cb.chain_len)) {
      fprintf(
          stderr, "[FATAL] Failed to build chain old graph: %s\n", cb.old_name);
      return 1;
    }
  }

  // 导出 context binary
  Qnn_ContextBinarySize_t bin_size = 0;
  if (api.contextGetBinarySize(ctx, &bin_size) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] contextGetBinarySize failed\n");
    return 1;
  }
  printf(
      "\n[INFO] Context binary size: %lu bytes (%.1f MB)\n",
      (unsigned long)bin_size,
      (double)bin_size / 1048576.0);

  std::vector<uint8_t> buffer(bin_size);
  Qnn_ContextBinarySize_t written = 0;
  if (api.contextGetBinary(ctx, buffer.data(), bin_size, &written) !=
      QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] contextGetBinary failed\n");
    return 1;
  }

  FILE* f = fopen(output_bin, "wb");
  if (!f) {
    fprintf(stderr, "[FATAL] Cannot open %s for writing\n", output_bin);
    return 1;
  }
  fwrite(buffer.data(), 1, (size_t)written, f);
  fclose(f);
  printf(
      "[INFO] Context binary saved: %s (%lu bytes)\n",
      output_bin,
      (unsigned long)written);

  // 保存 tensor ID 元数据 (设备端 --execute 模式需要)
  if (!save_graph_metadata(output_bin)) {
    fprintf(stderr, "[FATAL] Failed to save graph metadata\n");
    return 1;
  }

  api.contextFree(ctx, nullptr);
  api.backendFree(backend);
  if (log_handle)
    api.logFree(log_handle);
  dlclose(htp_handle);

  printf("\n========================================\n");
  printf("  Compilation complete!\n");
  printf("  Output: %s\n", output_bin);
  printf("  Next: push to device and run with --execute\n");
  printf("========================================\n");
  return 0;
}

//==============================================================================
// 测试与 Benchmark 执行逻辑 (execute 和 simulator 共用)
//==============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void check(const char* name, bool ok) {
  if (ok) {
    printf("  [PASS] %s\n", name);
    ++g_passed;
  } else {
    printf("  [FAIL] %s  <<<\n", name);
    ++g_failed;
  }
}

// 执行已编译的图: 使用 g_graph_tensors 中保存的 tensor (带正确 ID)
static bool execute_graph(
    const QnnApi& api,
    const char* graph_name,
    const fp16_t* a_data,
    const fp16_t* b_data,
    fp16_t* out_data,
    uint32_t total) {
  auto it = g_graph_tensors.find(graph_name);
  if (it == g_graph_tensors.end()) {
    fprintf(stderr, "[ERROR] Graph '%s' not found in tensor map\n", graph_name);
    return false;
  }
  GraphTensors& gt = it->second;
  uint32_t data_bytes = total * sizeof(fp16_t);

  // 复制 tensor 结构 (保留 ID、rank、dimensions 等), 只更新 clientBuf
  Qnn_Tensor_t t_a = gt.t_a;
  t_a.v1.clientBuf = {(void*)a_data, data_bytes};

  Qnn_Tensor_t t_b = gt.t_b;
  t_b.v1.clientBuf = {(void*)b_data, data_bytes};

  Qnn_Tensor_t t_out = gt.t_out;
  t_out.v1.clientBuf = {(void*)out_data, data_bytes};

  Qnn_Tensor_t inputs[] = {t_a, t_b};
  Qnn_Tensor_t outputs[] = {t_out};

  return api.graphExecute(gt.graph, inputs, 2, outputs, 1, nullptr, nullptr) ==
      QNN_SUCCESS;
}

static void run_correctness_tests(const QnnApi& api, Qnn_ContextHandle_t ctx) {
  printf("\n=== Correctness Tests ===\n");

  // TC-1: 1.0 + 2.0 = 3.0 [1,1,1,128]
  {
    constexpr uint32_t N = 128;
    std::vector<fp16_t> a(N, f32_to_f16(1.0f));
    std::vector<fp16_t> b(N, f32_to_f16(2.0f));
    std::vector<fp16_t> out(N, 0);
    if (execute_graph(api, "test_basic", a.data(), b.data(), out.data(), N)) {
      bool ok = true;
      for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(f16_to_f32(out[i]) - 3.0f) > 0.05f) {
          ok = false;
          break;
        }
      }
      check("TC-1  1.0 + 2.0 = 3.0 [1,1,1,128]", ok);
    } else {
      check("TC-1  1.0 + 2.0 = 3.0 [1,1,1,128]", false);
    }
  }

  // TC-2: 0.0 + 0.0 = 0.0 [1,1,1,64]
  {
    constexpr uint32_t N = 64;
    std::vector<fp16_t> a(N, f32_to_f16(0.0f));
    std::vector<fp16_t> b(N, f32_to_f16(0.0f));
    std::vector<fp16_t> out(N, f32_to_f16(99.0f));
    if (execute_graph(api, "test_zero", a.data(), b.data(), out.data(), N)) {
      bool ok = true;
      for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(f16_to_f32(out[i])) > 1e-3f) {
          ok = false;
          break;
        }
      }
      check("TC-2  0.0 + 0.0 = 0.0 [1,1,1,64]", ok);
    } else {
      check("TC-2  0.0 + 0.0 = 0.0 [1,1,1,64]", false);
    }
  }

  // TC-3: 1.5 + (-1.5) = 0.0 [1,1,1,64]
  {
    constexpr uint32_t N = 64;
    std::vector<fp16_t> a(N, f32_to_f16(1.5f));
    std::vector<fp16_t> b(N, f32_to_f16(-1.5f));
    std::vector<fp16_t> out(N, 0);
    if (execute_graph(api, "test_cancel", a.data(), b.data(), out.data(), N)) {
      bool ok = true;
      for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(f16_to_f32(out[i])) > 5e-3f) {
          ok = false;
          break;
        }
      }
      check("TC-3  1.5 + (-1.5) = 0.0 [1,1,1,64]", ok);
    } else {
      check("TC-3  1.5 + (-1.5) = 0.0 [1,1,1,64]", false);
    }
  }

  // TC-4: 0.5 + 0.25 = 0.75 [1,1,1,4096]
  {
    constexpr uint32_t N = 4096;
    std::vector<fp16_t> a(N, f32_to_f16(0.5f));
    std::vector<fp16_t> b(N, f32_to_f16(0.25f));
    std::vector<fp16_t> out(N, 0);
    if (execute_graph(api, "test_large", a.data(), b.data(), out.data(), N)) {
      bool ok = true;
      for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(f16_to_f32(out[i]) - 0.75f) > 0.02f) {
          ok = false;
          break;
        }
      }
      check("TC-4  0.5 + 0.25 = 0.75 [1,1,1,4096] (LLM hidden_dim)", ok);
    } else {
      check("TC-4  0.5 + 0.25 = 0.75 [1,1,1,4096]", false);
    }
  }

  // TC-5: Mixed values, N=100 (non-64-aligned, tail test)
  {
    constexpr uint32_t N = 100;
    std::vector<fp16_t> a(N), b(N), out(N, 0);
    for (uint32_t i = 0; i < N; i++) {
      a[i] = f32_to_f16(0.1f * (i % 10));
      b[i] = f32_to_f16(0.2f * (i % 5));
    }
    if (execute_graph(
            api, "test_nonaligned", a.data(), b.data(), out.data(), N)) {
      bool ok = true;
      for (uint32_t i = 0; i < N; i++) {
        float expected = f16_to_f32(a[i]) + f16_to_f32(b[i]);
        float actual = f16_to_f32(out[i]);
        if (std::fabs(actual - expected) > 0.1f) {
          printf(
              "    mismatch at [%u]: expected=%.4f actual=%.4f\n",
              i,
              expected,
              actual);
          ok = false;
          break;
        }
      }
      check("TC-5  Mixed values, N=100 (non-64-aligned, tail test)", ok);
    } else {
      check("TC-5  Mixed values, N=100 (non-64-aligned)", false);
    }
  }
}

// 测量单个图的 execute 延迟
static double measure_graph_us(
    const QnnApi& api,
    const char* name,
    const fp16_t* a,
    const fp16_t* b,
    fp16_t* out,
    uint32_t N,
    int warmup,
    int iters) {
  for (int w = 0; w < warmup; w++)
    execute_graph(api, name, a, b, out, N);
  double total = 0.0;
  for (int it = 0; it < iters; it++) {
    double t0 = now_us();
    if (!execute_graph(api, name, a, b, out, N))
      return -1.0;
    total += (now_us() - t0);
  }
  return total / iters;
}

static void run_benchmarks(const QnnApi& api, Qnn_ContextHandle_t ctx) {
  printf("\n=== Performance Benchmark ===\n");
  printf("  Background:\n");
  printf("    OLD path: TMANFinalize(Tcm->Main) -> [DMA: Tcm->Main]\n");
  printf("              -> q::Add(Main+Main->Main) -> [DMA: Main->Tcm]\n");
  printf("    NEW path: TMANFinalize(Tcm->Tcm)  -> TMANAdd(Tcm+Main->Tcm)\n");
  printf("  Eliminated: 2x Tcm<->MainMemory DMA per residual add\n");
  printf("  Note: DMA copies are HTP-compiler internal behavior,\n");
  printf("        not measurable via QNN API. Below compares op-level only.\n");
  printf("        Real-world savings include the 2x DMA overhead ON TOP.\n\n");

  for (int gi = 0; gi < g_num_bench_graphs; gi++) {
    const auto& gd = g_bench_graphs[gi];
    uint32_t N = gd.batch * gd.height * gd.width * gd.channels;
    uint32_t data_bytes = N * sizeof(fp16_t);

    printf(
        "--- Bench: [%u,%u,%u,%u]  N=%u  (%.1f KB) ---\n",
        gd.batch,
        gd.height,
        gd.width,
        gd.channels,
        N,
        (double)data_bytes / 1024.0);

    if (g_graph_tensors.find(gd.name) == g_graph_tensors.end()) {
      printf("  [SKIP] TMANAdd graph not found: %s\n", gd.name);
      continue;
    }

    std::vector<fp16_t> a(N), b(N), out(N);
    for (uint32_t i = 0; i < N; i++) {
      a[i] = f32_to_f16(0.1f * (float)(i % 10) - 0.5f);
      b[i] = f32_to_f16(0.05f * (float)(i % 20) - 0.3f);
    }

    const int warmup = 10;
    const int iters = 100;

    // NEW: TMANAdd (Tcm a + Main b -> Tcm)
    double tman_us = measure_graph_us(
        api, gd.name, a.data(), b.data(), out.data(), N, warmup, iters);
    if (tman_us < 0) {
      printf("  [ERROR] TMANAdd execute failed\n");
      continue;
    }

    // OLD: ElementWiseAdd (Main a + Main b -> Main)
    double elemAdd_us = -1.0;
    bool has_old = gd.has_old_graph && gd.old_name &&
        g_graph_tensors.find(gd.old_name) != g_graph_tensors.end();
    if (has_old) {
      elemAdd_us = measure_graph_us(
          api, gd.old_name, a.data(), b.data(), out.data(), N, warmup, iters);
    }

    printf("\n");
    printf("  [NEW] TcmRAdd (Tcm+Main->Tcm):  %8.1f us/iter\n", tman_us);
    if (elemAdd_us > 0) {
      printf("  [OLD] QNNAdd (Main+Main->Main): %7.1f us/iter\n", elemAdd_us);
      printf("\n");
      printf("  Op-level speedup:              %.2fx\n", elemAdd_us / tman_us);
      double saved_us = elemAdd_us - tman_us;
      printf(
          "  Op-level saved:                %8.1f us (%.1f%%)\n",
          saved_us,
          saved_us / elemAdd_us * 100.0);
      printf("\n");
      printf(
          "  * Plus 2x DMA (Tcm<->Main) eliminated per add (not measured).\n");
      printf(
          "    DMA cost estimate: ~%.1f KB * 2 roundtrips.\n",
          (double)data_bytes / 1024.0);
      printf("    On SM8650 HTP v75 (DDR ~34 GB/s):\n");
      double dma_est_us = (double)data_bytes / (34.0e9 / 1e6);
      printf(
          "      ~1x DMA ~ %.1f us  (2x = %.1f us)\n",
          dma_est_us,
          2.0 * dma_est_us);
      double total_saved = saved_us + 2.0 * dma_est_us;
      printf("    Estimated total saved/add:   %.1f us\n", total_saved);
      printf("\n");
      // Qwen3-1.7B: 28 layers, ~2 residual adds per layer = 56
      printf("  Qwen3-1.7B extrapolation (28 layers, 2 adds/layer = 56):\n");
      printf(
          "    Saved per token (op only):    ~%.0f us (%.2f ms)\n",
          saved_us * 56.0,
          saved_us * 56.0 / 1000.0);
      printf(
          "    Saved per token (op+DMA est): ~%.0f us (%.2f ms)\n",
          total_saved * 56.0,
          total_saved * 56.0 / 1000.0);
    }
    printf("\n");
  }
}

//==============================================================================
// Pipeline Benchmark: 链式图, 中间数据留在 TCM, 消除人为 DMA 开销
//==============================================================================
static void run_pipeline_benchmarks(
    const QnnApi& api,
    Qnn_ContextHandle_t ctx) {
  printf("\n=== Pipeline Benchmark (Chained Ops) ===\n");
  printf("  Simulates real LLM inference pipeline:\n");
  printf("    K ops chained in ONE graph, intermediate data stays in TCM\n");
  printf("    Only 1st op pays DMA(clientBuf->TCM), the rest reuse TCM data\n");
  printf("    Per-op time = total_graph_time / K\n");
  printf(
      "    This eliminates the unfair DMA penalty in isolated benchmarks.\n");
  printf("\n");

  for (int ci = 0; ci < g_num_chain_bench; ci++) {
    const auto& cb = g_chain_bench[ci];
    uint32_t N = cb.batch * cb.height * cb.width * cb.channels;
    uint32_t data_bytes = N * sizeof(fp16_t);

    printf(
        "--- Pipeline: [%u,%u,%u,%u]  N=%u  (%.1f KB)  chain=%d ops ---\n",
        cb.batch,
        cb.height,
        cb.width,
        cb.channels,
        N,
        (double)data_bytes / 1024.0,
        cb.chain_len);

    if (g_graph_tensors.find(cb.name) == g_graph_tensors.end()) {
      printf("  [SKIP] Chain graph not found: %s\n\n", cb.name);
      continue;
    }

    std::vector<fp16_t> a(N), b(N), out(N);
    for (uint32_t i = 0; i < N; i++) {
      a[i] = f32_to_f16(0.1f * (float)(i % 10) - 0.5f);
      b[i] = f32_to_f16(0.05f * (float)(i % 20) - 0.3f);
    }

    const int warmup = 10;
    const int iters = 100;

    // TMANAdd chain
    double tman_total = measure_graph_us(
        api, cb.name, a.data(), b.data(), out.data(), N, warmup, iters);
    double tman_per_op = (tman_total > 0) ? tman_total / cb.chain_len : -1.0;

    // ElemAdd chain
    double elem_total = -1.0, elem_per_op = -1.0;
    if (g_graph_tensors.find(cb.old_name) != g_graph_tensors.end()) {
      elem_total = measure_graph_us(
          api, cb.old_name, a.data(), b.data(), out.data(), N, warmup, iters);
      elem_per_op = (elem_total > 0) ? elem_total / cb.chain_len : -1.0;
    }

    printf("\n");
    printf(
        "  [NEW] TcmRAdd chain (%2d ops): total %8.1f us  per-op %7.1f us\n",
        cb.chain_len,
        tman_total,
        tman_per_op);
    if (elem_total > 0) {
      printf(
          "  [OLD] QNNAdd chain (%2d ops): total %8.1f us  per-op %7.1f us\n",
          cb.chain_len,
          elem_total,
          elem_per_op);
      printf("\n");
      if (tman_per_op > 0 && elem_per_op > 0) {
        double speedup = elem_per_op / tman_per_op;
        double saved = elem_per_op - tman_per_op;
        printf("  Per-op speedup:  %.2fx\n", speedup);
        printf(
            "  Per-op saved:    %.1f us (%.1f%%)\n",
            saved,
            saved / elem_per_op * 100.0);
        printf("\n");
        printf("  Qwen3-1.7B extrapolation (56 adds/token):\n");
        printf(
            "    Saved per token: ~%.0f us (%.2f ms)\n",
            saved * 56.0,
            saved * 56.0 / 1000.0);
      }
    }
    printf("\n");
  }
}

//==============================================================================
// 模式 2: 设备执行 (--execute)
//==============================================================================
static int mode_execute(
    const char* htp_lib,
    const char* pkg_lib,
    const char* context_bin) {
  printf("========================================\n");
  printf("  TMANAdd On-Device Test & Benchmark\n");
  printf("========================================\n\n");

  // 加载 context binary
  FILE* f = fopen(context_bin, "rb");
  if (!f) {
    fprintf(stderr, "[FATAL] Cannot open context binary: %s\n", context_bin);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> bin_data(fsize);
  if ((long)fread(bin_data.data(), 1, fsize, f) != fsize) {
    fprintf(stderr, "[FATAL] Failed to read context binary\n");
    fclose(f);
    return 1;
  }
  fclose(f);
  printf(
      "[INFO] Loaded context binary: %s (%ld bytes, %.1f MB)\n",
      context_bin,
      fsize,
      (double)fsize / 1048576.0);

  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api))
    return 1;

  Qnn_LogHandle_t log_handle = nullptr;
  api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle);

  Qnn_BackendHandle_t backend = nullptr;
  if (api.backendCreate(log_handle, nullptr, &backend) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] backendCreate failed\n");
    return 1;
  }

  // Device (HTP v75)
  QnnHtpDevice_CustomConfig_t htpDevConfig{};
  htpDevConfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
  htpDevConfig.arch.deviceId = 0;
  htpDevConfig.arch.arch = QNN_HTP_DEVICE_ARCH_V75;

  QnnDevice_Config_t devConfig;
  devConfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
  devConfig.customConfig = &htpDevConfig;

  const QnnDevice_Config_t* devConfigs[] = {&devConfig, nullptr};
  Qnn_DeviceHandle_t device = nullptr;
  if (api.deviceCreate(log_handle, devConfigs, &device) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] deviceCreate failed\n");
    return 1;
  }
  printf("[INFO] Device created (HTP v75)\n");

  // 注册 op package: DSP 侧执行内核 (hexagon .so)
  Qnn_ErrorHandle_t err = api.registerOpPackage(
      backend, pkg_lib, "TMANOpPackageInterfaceProvider", "HTP");
  if (err != QNN_SUCCESS) {
    fprintf(
        stderr,
        "[FATAL] registerOpPackage failed: 0x%lx\n",
        (unsigned long)err);
    return 1;
  }
  printf("[INFO] Op package registered for DSP: %s\n", pkg_lib);

  // 从 binary 恢复 context (包含所有已编译的图)
  Qnn_ContextHandle_t ctx = nullptr;
  err = api.contextCreateFromBinary(
      backend,
      device,
      nullptr,
      bin_data.data(),
      (Qnn_ContextBinarySize_t)bin_data.size(),
      &ctx,
      nullptr);
  if (err != QNN_SUCCESS) {
    fprintf(
        stderr,
        "[FATAL] contextCreateFromBinary failed: 0x%lx\n",
        (unsigned long)err);
    return 1;
  }
  printf("[INFO] Context created from binary\n");

  // 从元数据恢复 tensor IDs + graphRetrieve 获取图句柄
  if (!load_graph_metadata_and_retrieve(api, ctx, context_bin)) {
    fprintf(stderr, "[FATAL] Failed to load graph metadata\n");
    return 1;
  }

  // 运行测试和 benchmark
  run_correctness_tests(api, ctx);
  printf("\nCorrectness: %d passed, %d failed\n", g_passed, g_failed);

  run_benchmarks(api, ctx);
  run_pipeline_benchmarks(api, ctx);

  printf("========================================\n");
  printf("  Final: %d passed, %d failed\n", g_passed, g_failed);
  printf(
      "  %s\n", g_failed == 0 ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");
  printf("========================================\n");

  api.contextFree(ctx, nullptr);
  api.deviceFree(device);
  api.backendFree(backend);
  if (log_handle)
    api.logFree(log_handle);
  dlclose(htp_handle);

  return (g_failed > 0) ? 1 : 0;
}

//==============================================================================
// 模式 3: 模拟器一体化 (x86)
//==============================================================================
static int mode_simulator(const char* htp_lib, const char* pkg_lib) {
  printf("========================================\n");
  printf("  TMANAdd Simulator Test & Benchmark\n");
  printf("========================================\n\n");

  void* htp_handle = nullptr;
  QnnApi api{};
  if (!load_qnn_api(htp_lib, &htp_handle, &api))
    return 1;

  Qnn_LogHandle_t log_handle = nullptr;
  api.logCreate(qnn_log_callback, QNN_LOG_LEVEL_WARN, &log_handle);

  Qnn_BackendHandle_t backend = nullptr;
  if (api.backendCreate(log_handle, nullptr, &backend) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] backendCreate failed\n");
    return 1;
  }

  if (api.registerOpPackage(
          backend, pkg_lib, "TMANOpPackageInterfaceProvider", nullptr) !=
      QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] registerOpPackage failed\n");
    return 1;
  }
  printf("[INFO] Op package registered: %s\n", pkg_lib);

  Qnn_ContextHandle_t ctx = nullptr;
  if (api.contextCreate(backend, nullptr, nullptr, &ctx) != QNN_SUCCESS) {
    fprintf(stderr, "[FATAL] contextCreate failed\n");
    return 1;
  }

  for (int i = 0; i < g_num_test_graphs; i++) {
    if (!build_tman_graph(api, ctx, g_test_graphs[i]))
      return 1;
  }
  for (int i = 0; i < g_num_bench_graphs; i++) {
    if (!build_tman_graph(api, ctx, g_bench_graphs[i]))
      return 1;
    if (!build_old_graph(api, ctx, g_bench_graphs[i]))
      return 1;
  }
  for (int i = 0; i < g_num_chain_bench; i++) {
    const auto& cb = g_chain_bench[i];
    if (!build_chained_graph(
            api,
            ctx,
            cb.name,
            "TMANOpPackage",
            "TMANAdd",
            "ca",
            "cb",
            "cout",
            cb.batch,
            cb.height,
            cb.width,
            cb.channels,
            cb.chain_len))
      return 1;
    if (!build_chained_graph(
            api,
            ctx,
            cb.old_name,
            "qti.aisw",
            "ElementWiseAdd",
            "old_ca",
            "old_cb",
            "old_cout",
            cb.batch,
            cb.height,
            cb.width,
            cb.channels,
            cb.chain_len))
      return 1;
  }
  run_correctness_tests(api, ctx);
  printf("\nCorrectness: %d passed, %d failed\n", g_passed, g_failed);

  run_benchmarks(api, ctx);
  run_pipeline_benchmarks(api, ctx);

  printf("========================================\n");
  printf("  Final: %d passed, %d failed\n", g_passed, g_failed);
  printf(
      "  %s\n", g_failed == 0 ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");
  printf("========================================\n");

  api.contextFree(ctx, nullptr);
  api.backendFree(backend);
  if (log_handle)
    api.logFree(log_handle);
  dlclose(htp_handle);

  return (g_failed > 0) ? 1 : 0;
}

//==============================================================================
// main
//==============================================================================
static void usage(const char* prog) {
  printf("TMANAdd Unit Test & Benchmark\n\n");
  printf("Usage:\n");
  printf("  模拟器模式 (x86 一体化):\n");
  printf("    %s <libQnnHtp.so> <op_pkg.so>\n\n", prog);
  printf("  离线编译 (x86 host → context binary):\n");
  printf(
      "    %s --compile <libQnnHtp.so> <x86_op_pkg.so> <output.bin>\n\n", prog);
  printf("  设备执行 (aarch64, 从 context binary 恢复):\n");
  printf(
      "    %s --execute <libQnnHtp.so> <hex_op_pkg.so> <context.bin>\n\n",
      prog);
  printf("典型工作流:\n");
  printf("  1. PC 编译 context binary:\n");
  printf("     %s --compile libQnnHtp.so x86_pkg.so tman_ctx.bin\n", prog);
  printf("  2. Push 到设备 + 执行:\n");
  printf(
      "     %s --execute ./libQnnHtp.so ./hex_pkg.so ./tman_ctx.bin\n\n", prog);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  std::string arg1 = argv[1];

  if (arg1 == "--compile") {
    if (argc < 5) {
      fprintf(stderr, "Error: --compile requires 3 arguments\n\n");
      usage(argv[0]);
      return 1;
    }
    return mode_compile(argv[2], argv[3], argv[4]);
  }

  if (arg1 == "--execute") {
    if (argc < 5) {
      fprintf(stderr, "Error: --execute requires 3 arguments\n\n");
      usage(argv[0]);
      return 1;
    }
    return mode_execute(argv[2], argv[3], argv[4]);
  }

  // 默认: 模拟器模式
  return mode_simulator(argv[1], argv[2]);
}
