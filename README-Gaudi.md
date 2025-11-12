## Run Docker Image
Use the following commands to run a Docker image.
```bash
$ docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

## Install MinerU from source
```bash
$ git clone https://gitee.com/intel-china/aisolution-mineru.git
$ git checkout release-2.5.4-hpu
$ pip install -e .[core]
```

## vlm-vllm-engine backend

### For vlm-vllm-engine backend: Install vllm from source
```bash
$ git clone https://github.com/HabanaAI/vllm-fork.git -b aice/v1.22.0
$ pip install -e .
$ git clone https://github.com/vllm-project/vllm.git
$ cp -r vllm/vllm/v1/sample/logits_processor vllm-fork/vllm/v1/sample/logits_processor
```

### Runing vlm-vllm-engine backend on Gaudi

#### Set environment variables
```bash
#!/bin/bash
export MAX_NUM_SEQS=16
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=True
export VLLM_GRAPH_RESERVED_MEM=0.5
export VLLM_GRAPH_PROMPT_RATIO=0.4
export VLLM_MULTIMODAL_BUCKETS="64,192,384,512,640,768,896,1024,1152,1280,1408,1536,1664,2496, 3136, 4096, 5504, 6272, 7104, 8192, 9216"
export MINERU_MODEL_SOURCE=local
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=0
export VLLM_FP32_SOFTMAX=true
export VLLM_FP32_SOFTMAX_VISION=true
```

#### Quick Usage via Command Line
```bash
mineru -p <input_path> -o <output_path>  -b vlm-vllm-engine 
```

#### Using http-client/server method

##### start vllm server and api server
```bash
# Start vllm server at port 30000 
mineru-vllm-server --host 0.0.0.0 --port 30000 --gpu-memory-utilization 0.7 2>&1 | tee -a server.log >/dev/null &
```
##### using minerU CLI 
```bash
export MINERU_VL_SERVER=http://0.0.0.0:30000
mineru -p "input.pdf"  -o "output" -b vlm-http-client -u ${MINERU_VL_SERVER}
```

##### using minerU API 
```bash
# claim mineru-vllm-server url to api server usgae
export MINERU_VL_SERVER=http://0.0.0.0:30000
# start MinerU API server at port 8007 
mineru-api --host 0.0.0.0 --port 8007 2>&1 | tee -a api.log >/dev/null &
```
```bash
curl -vvv -X POST "http://0.0.0.0:8007/file_parse" \
  -H "Accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/data/test.pdf;type=application/pdf" \
  -F "output_dir=./out" \
  -F "lang_list[0]=zh" \
  -F "backend=vlm-http-client" \
  -F "parse_method=ocr" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "server_url=http://0.0.0.0:30000" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_model_output=true" \
  -F "return_content_list=true" \
  -F "return_images=true" \
  -F "response_format_zip=true" \
  -F "start_page_id=1" \
  -F "end_page_id=10" \
  --output result.zip
```



## Pipeline backend

### For Pipeline backend: Install optimum-habana from source
```bash
$ git clone https://github.com/huggingface/optimum-habana
$ cd optimum-habana && git checkout 48a2dae1709b50630c6fc93fdf76c52fdfb82566
$ pip install -e .
```


### Runing pipeline on Gaudi

#### modify code in doclayout_yolo and ultralytics for hpu
```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/engine/predictor.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/engine/predictor.py
```
Change the device parameter in setup_model() as following:

```bash
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        if self.args.device == "hpu":
            device = self.args.device
        else:
            device = select_device(self.args.device, verbose=verbose)
        self.model = AutoBackend(
            weights=model or self.args.model,
            #device=select_device(self.args.device, verbose=verbose),
            device=device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=False,
            verbose=verbose,
        )
```

```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/nn/autobackend.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/nn/autobackend.py
```
Add the  hpu device branch in warmup() as following:
```bash
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        #if any(warmup_types) and (self.device.type != "cpu" or self.triton):
        if self.device  == "hpu":
            im = torch.empty(*imgsz, dtype=torch.bfloat16 if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup
        elif any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup
```

#### Runing pipeline on hpu
```bash
$ MINERU_DEVICE_MODE=hpu mineru -p ./test.pdf -o ./ -d hpu  -b pipeline -m ocr
```