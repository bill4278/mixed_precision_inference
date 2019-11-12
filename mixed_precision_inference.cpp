#include <NvInfer.h>
#include <cuda_runtime_api.h> 
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

using namespace nvinfer1;

class Logger : public ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {   
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }   

    Severity reportableSeverity{Severity::kINFO};
};

static Logger gLogger(ILogger::Severity::kINFO);

static INetworkDefinition* gNET = nullptr;
static DataType gDATA_TYPE = DataType::kINT8;
static const char* kINPUT_NAME = "my_intput";
static const char* kOUTPUT_NAME = "my_output";
static const char* kWEIGHT_NAME = "my_weight";
static int gMAX_BATCH_SIZE = 8;
static size_t gWORKSAPCE_SIZE = 0;
static int gCUDA_DEVICE = 0;
static std::vector<float*> gWEIGHT;
static const int kINPUT_CHANNEL = 256;
static const int kINPUT_SIZE = 128;
static const int kOUTPUT_CHANNEL = 256;
static const int kFILTER_SIZE = 3;

void DelWeight(void) {
  for (auto* ptr : gWEIGHT) {
    delete [] ptr;
  }
}

void SetTensorRange(ITensor* input_tensor) {
  input_tensor->setDynamicRange(-1.0, 1.0);
}

void SetLayerPrecision(ILayer* layer, DataType dtype = DataType::kFLOAT) {
  layer->setPrecision(dtype);
  layer->setOutputType(0, dtype);

  if (dtype == DataType::kINT8) {
    SetTensorRange(layer->getOutput(0));
  }
}

ITensor* Input(const char* input_name, int c, int hw) {
  auto* input_out = gNET->addInput(input_name, DataType::kFLOAT, DimsCHW(c, hw, hw));

  if (gDATA_TYPE == DataType::kINT8) {
    SetTensorRange(input_out);
  }
  return input_out;
}

ITensor* Input8(const char* input_name, int c, int hw) {
  auto* input_out = gNET->addInput(input_name, DataType::kINT8, DimsCHW(c, hw, hw));

  if (gDATA_TYPE == DataType::kINT8) {
    SetTensorRange(input_out);
  }
  return input_out;
}

ITensor* Conv(ITensor* input_tensor, int fco, int fci, int fhw, int stride, DataType dtype) {
  int conv_w_num = fco * fci * fhw * fhw;
  float* conv_w_data = new float[conv_w_num]();
  gWEIGHT.push_back(conv_w_data);
  Weights weight{DataType::kFLOAT, static_cast<void*>(conv_w_data), static_cast<int64_t>(conv_w_num)};

  float* bias_data = new float[fco]();
  gWEIGHT.push_back(bias_data);
  Weights bias{DataType::kFLOAT, static_cast<void*>(bias_data), static_cast<int64_t>(fco)};

  DimsHW nv_ksize(fhw, fhw);

  auto *iconv_layer = gNET->addConvolution(*input_tensor, fco, nv_ksize, weight, bias);
  DimsHW nv_dilations(1, 1);
  DimsHW nv_strides(stride, stride);
  DimsHW nv_paddings(fhw/2, fhw/2);
  iconv_layer->setDilation(nv_dilations);
  iconv_layer->setStride(nv_strides);
  iconv_layer->setPadding(nv_paddings);
  iconv_layer->setNbGroups(1);

  static int count = 0;
  std::string name = "my_conv" + std::to_string(count);
  iconv_layer->setName(name.c_str());

  auto tensor_out = iconv_layer->getOutput(0);
  name += std::string("_output");
  tensor_out->setName(name.c_str());
  count++;
  
  SetLayerPrecision(iconv_layer, dtype);

  return tensor_out;
}

ITensor* ElemAdd(ITensor* input_tensor1, ITensor* input_tensor2, DataType dtype) {
  auto* elem_add_layer = gNET->addElementWise(*input_tensor1, *input_tensor2, ElementWiseOperation::kSUM);

  // SetLayerPrecision(elem_add_layer);
  // force layer to be FP32
  elem_add_layer->setPrecision(DataType::kFLOAT);
  elem_add_layer->setOutputType(0, DataType::kFLOAT);
  return elem_add_layer->getOutput(0);
}

std::vector<void*> PrepareInputOutputBuffers(const ICudaEngine* engine, 
  int max_batch_size,
  const char* output_name, int co,
  const char* input_name, int ci, int hw,
  const char* weight_name) {

  std::vector<void*> buffers(3);

  int input_num = max_batch_size * ci * hw * hw;
  
  float *input_dev; 
  cudaMalloc((void**)&input_dev, input_num * sizeof(float));

  const int input_index = engine->getBindingIndex(input_name);
  buffers[input_index] = static_cast<void*>(input_dev);

  int output_num = max_batch_size * co * hw * hw;
  float *output_dev;
  cudaMalloc((void**)&output_dev, output_num * sizeof(float));
  const int output_index = engine->getBindingIndex(output_name);
  buffers[output_index] = static_cast<void*>(output_dev);

  int weight_num = max_batch_size * co * hw * hw;
  float *weight_dev;
  cudaMalloc((void**)&weight_dev, weight_num * sizeof(float));
  const int weight_index = engine->getBindingIndex(weight_name);
  buffers[weight_index] = static_cast<void*>(weight_dev);

  return buffers;
}

void Inference(IExecutionContext* context, std::vector<void*> buffers, cudaStream_t stream, int max_batch_size) {
  struct timeval start;
  struct timeval end; 

  for (int j = 0; j < 10; ++j) {
    gettimeofday(&start, nullptr);
    for (int i = 0; i < 10; ++i) {
      context->enqueue(1, buffers.data(), stream, nullptr);
    }
    cudaStreamSynchronize(stream);
    gettimeofday(&end, nullptr);
    unsigned long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    float avg_delta = static_cast<float>(diff) / 10.f;
    std::cout << "time (ms): " << (avg_delta / 1000.f) <<
      ", FPS: " << (static_cast<float>(max_batch_size) / (avg_delta / 1000000.f)) << std::endl;
  }
}

void UseTrtLayerAPI(int max_batch_size, size_t workspace_size, DataType exec_type) {
  auto builder = createInferBuilder(gLogger);
  gNET = builder->createNetwork();

  auto* input    = Input(kINPUT_NAME, kINPUT_CHANNEL, kINPUT_SIZE);
  auto* conv0 = Conv(input, kOUTPUT_CHANNEL, kINPUT_CHANNEL, kFILTER_SIZE, 1, exec_type);
  auto* conv1 = Conv(conv0, kOUTPUT_CHANNEL, kINPUT_CHANNEL, kFILTER_SIZE, 1, exec_type);
  auto* conv2 = Conv(conv1, kOUTPUT_CHANNEL, kINPUT_CHANNEL, kFILTER_SIZE, 1, exec_type);
  auto* w_out    = Input(kWEIGHT_NAME, kOUTPUT_CHANNEL, kINPUT_SIZE);
  auto* output   = ElemAdd(conv2, w_out, DataType::kFLOAT);
  output->setName(kOUTPUT_NAME);
  gNET->markOutput(*output);

  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(workspace_size);

  if (exec_type == DataType::kHALF) {
    builder->setFp16Mode(true);
  }
  else if (exec_type == DataType::kINT8) {
    builder->setInt8Mode(true);
    builder->setInt8Calibrator(nullptr);
  }

  builder->setStrictTypeConstraints(true);
  auto engine = builder->buildCudaEngine(*gNET); 
  gNET->destroy(); 
  DelWeight();

  auto buffers = PrepareInputOutputBuffers(engine, max_batch_size,
    kOUTPUT_NAME, kOUTPUT_CHANNEL,
    kINPUT_NAME, kINPUT_CHANNEL, kINPUT_SIZE,
    kWEIGHT_NAME);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::cout << "Executing..." << std::endl;
  auto context = engine->createExecutionContext();

  Inference(context, buffers, stream, max_batch_size);

  /*
  auto serial = engine->serialize();
  std::ofstream outfile ("serial.bin",std::ofstream::binary);
  outfile.write((const char*)serial->data(), serial->size());
  outfile.close();
  serial->destroy();
  */
  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
}

void ParseArguments(int argc, const char** argv) {
  std::cout << "Usage:\n\tmax batch size, workspace size, type, device\n\n" <<
               "\tworkspace size = input * 1 MB\n" <<
               "\ttype:\n\t\tset FP32 if input == 0"
                    "\n\t\tset FP16 if input == 1"
                    "\n\t\tset INT8 if input == 2\n" << std::endl;

  if (argc > 1) {
    gMAX_BATCH_SIZE = std::atoi(argv[1]);
  }

  if (argc > 2) {
    gWORKSAPCE_SIZE = static_cast<size_t>(1024*1024) * static_cast<size_t>(std::atoi(argv[2]));
  }

  std::map<int, DataType> int2type = {
    {0, DataType::kFLOAT},
    {1, DataType::kHALF},
    {2, DataType::kINT8}
  };

  if (argc > 3) {
    int idx = std::atoi(argv[3]);
    if (idx >= 0 && idx <= 2) {
      gDATA_TYPE = int2type[idx];
    }
    else {
      std::cerr << "invalid type arguments" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::map<DataType, std::string> type2str = {
    {DataType::kFLOAT, "FP32"},
    {DataType::kHALF,  "FP16"},
    {DataType::kINT8,  "INT8"}
  };

  std::cout << "Summary: "
            << "\n\tmax batch size: " << gMAX_BATCH_SIZE
            << "\n\tworkspace size: " << gWORKSAPCE_SIZE 
            << "\n\ttype: " << type2str[gDATA_TYPE] << std::endl;

  if (argc > 4) {
    gCUDA_DEVICE = std::atoi(argv[4]);
  }
  assert(cudaSetDevice(gCUDA_DEVICE) == cudaSuccess);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gCUDA_DEVICE);
  std::cout << "\tdevice: " << gCUDA_DEVICE << ", " << prop.name << std::endl << std::endl;
}

int main(int argc, const char** argv) {
  ParseArguments(argc, argv);
  UseTrtLayerAPI(gMAX_BATCH_SIZE, gWORKSAPCE_SIZE, gDATA_TYPE);
  return 0;
}
