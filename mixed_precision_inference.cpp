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
static int gMAX_BATCH_SIZE = 2;
static size_t gWORKSAPCE_SIZE = 0;
static int gCUDA_DEVICE = 0;
static std::vector<float*> gWEIGHT;
static const int kINPUT_CHANNEL = 3;
static const int kINPUT_SIZE = 8;
static const int kOUTPUT_CHANNEL = 3;

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

ITensor* Conv(ITensor* input_tensor, int fco, int fci, int fhw, 
  float init_file_value, float init_bias_value, DataType dtype) {

  int w_count = fco * fci * fhw * fhw;
  float* w_data = new float[w_count]();
  for (int i = 0; i < w_count; ++i) {
    w_data[i] = init_file_value;
  }
  gWEIGHT.push_back(w_data);
  Weights weight{DataType::kFLOAT, static_cast<void*>(w_data), static_cast<int64_t>(w_count)};

  float* bias_data = new float[fco]();
  for (int i = 0; i < fco; ++i) {
    bias_data[i] = init_bias_value;
  }
  gWEIGHT.push_back(bias_data);
  Weights bias{DataType::kFLOAT, static_cast<void*>(bias_data), static_cast<int64_t>(fco)};

  DimsHW nv_ksize(fhw, fhw);

  auto *conv_layer = gNET->addConvolution(*input_tensor, fco, nv_ksize, weight, bias);
  DimsHW nv_dilations(1, 1);
  DimsHW nv_strides(1, 1);
  DimsHW nv_paddings(fhw/2, fhw/2);
  conv_layer->setDilation(nv_dilations);
  conv_layer->setStride(nv_strides);
  conv_layer->setPadding(nv_paddings);
  conv_layer->setNbGroups(1);

  static int count = 0;
  std::string name = "my_conv" + std::to_string(count);
  conv_layer->setName(name.c_str());

  auto tensor_out = conv_layer->getOutput(0);
  name += std::string("_output");
  tensor_out->setName(name.c_str());
  count++;
  
  SetLayerPrecision(conv_layer, dtype);

  return tensor_out;
}

ITensor* Concat(ITensor* input_tensor1, ITensor* input_tensor2, DataType dtype)
{
  ITensor* inputs[2] = {input_tensor1, input_tensor2};
  auto* concat_layer = gNET->addConcatenation(inputs, 2);
  
  SetLayerPrecision(concat_layer, dtype);

  return concat_layer->getOutput(0);
}

int GetDimsCount(Dims dims) {
  int count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    count *= dims.d[i];
  }
  return count;  
}

void VerifyDims(Dims dims) {
  printf("Dims: ");
  for (int i = 0; i < dims.nbDims; ++i) {
    printf("%d, ", dims.d[i]);
  }
  printf("\n");
}

std::vector<void*> PrepareInputOutputBuffers(const ICudaEngine* engine, 
  int max_batch_size, const char* output_name, const char* input_name) {

  auto nbBindings = engine->getNbBindings();
  std::vector<void*> buffers(nbBindings);

  auto input_index = engine->getBindingIndex(input_name);
  auto input_dims = engine->getBindingDimensions(input_index);
  VerifyDims(input_dims);
  auto input_count = GetDimsCount(input_dims);

  float *d_input; 
  cudaMalloc((void**)&d_input, max_batch_size * input_count * sizeof(float));
  buffers[input_index] = static_cast<void*>(d_input);
  auto *h_input = new float[max_batch_size * input_count];
  auto *base = h_input;
  for (int b = 0; b < max_batch_size; ++b) {
    for (int c = 0; c < input_count; ++c) {
      *h_input = static_cast<float>(b + 1);
      h_input++;
    }
  }
  cudaMemcpy(d_input, base, max_batch_size * input_count * sizeof(float), cudaMemcpyHostToDevice);
  delete [] base;


  const int output_index = engine->getBindingIndex(output_name);
  auto output_dims = engine->getBindingDimensions(output_index);
  VerifyDims(output_dims);
  auto output_count = GetDimsCount(output_dims);

  float *d_output;
  cudaMalloc((void**)&d_output, max_batch_size * output_count * sizeof(float));
  buffers[output_index] = static_cast<void*>(d_output);

  return buffers;
}

void VerifyOutputTensor(void* d_ptr, int max_batch_size, Dims output_dims) {
  assert(output_dims.nbDims == 3);
  assert(output_dims.d[1] == output_dims.d[2]);
  int co = output_dims.d[0];
  int hw = output_dims.d[1];
  int count = max_batch_size * co * hw * hw; 
  auto* h_ptr = new float[count]();
  auto* base_ptr = h_ptr;
  cudaMemcpy(h_ptr, d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);

  for (int b = 0; b < max_batch_size; ++b) {
    printf("batch %d:\n", b);
    for (int c = 0; c < co; ++c) {
      printf("channel %d:\n", c);
      for (int y = 0; y < hw; ++y) {
        for (int x = 0; x < hw; ++x) {
          printf("%.2f, ", *h_ptr);
          h_ptr++;
        }
        printf("\n");
      }
      printf("\n\n");
    }
    printf("\n\n\n");
  }

  delete [] base_ptr;
}

void Inference(IExecutionContext* context, std::vector<void*> buffers,
  int max_batch_size, int output_index) {

  context->execute(max_batch_size, buffers.data());
  auto output_dims = context->getBindingDimensions(1);
  VerifyOutputTensor(buffers[output_index], max_batch_size, output_dims);    
}

void UseTrtLayerAPI(int max_batch_size, size_t workspace_size, DataType exec_type) {
  auto builder = createInferBuilder(gLogger);
  gNET = builder->createNetwork();

  auto* input  = Input(kINPUT_NAME, kINPUT_CHANNEL, kINPUT_SIZE);
  
  auto* conv0  = Conv(input, kOUTPUT_CHANNEL, kINPUT_CHANNEL, 1, 1.0, 0.0, exec_type);
  auto* conv2  = Conv(input, kOUTPUT_CHANNEL, kINPUT_CHANNEL, 1, 2.0, 0.0, exec_type);
  auto* output = Concat(conv0, conv2, exec_type);

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
    kOUTPUT_NAME, kINPUT_NAME);

  std::cout << "Executing..." << std::endl;
  auto context = engine->createExecutionContext();
  auto output_index = engine->getBindingIndex(kOUTPUT_NAME);

  Inference(context, buffers, max_batch_size, output_index);

  context->destroy();
  engine->destroy();

  for (auto* ptr : buffers) {
    cudaFree(ptr);
  }
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
