
## Install MinerU from source
```bash
git clone https://github.com/opendatalab/MinerU.git
git checkout release-1.3.8
pip install -e .[full]
```

## Download models
```bash
cd MinerU/projects/web_api
python download_models.py
```
Default downloaded moodel_dir is: /opt/models
layoutreader_model_dir is: /opt/layoutreader


## Runing pipeline on CPU
```bash
MINERU_TOOLS_CONFIG_JSON=/home/MinerU/gaudi/magic-pdf.json magic-pdf -p ./test.pdf -o ./  -m ocr
```

## Runing pipeline on Gaudi

### modify code in doclayout_yolo and ultralytics for hpu
```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/engine/predictor.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/engine/predictor.py
```
Change the device parameter in setup_model() as following:

![alt text](image.png)

```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/nn/autobackend.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/nn/autobackend.py
```
Add the  hpu device branch in warmup() as following:

![alt text](image-1.png)

### Runing pipeline on hpu
You can chage the param in "magic-pdf_hpu.json"
As Unimernet_small has not enabled on gaudi yet， So "formula-config" is set to "enable": false.
As slanet_plus has not enabled on gaudi yet， So "table-config" is set to "enable": false.
```bash
MINERU_TOOLS_CONFIG_JSON=/home/MinerU/gaudi/magic-pdf_hpu.json magic-pdf -p ./test.pdf -o ./  -m ocr
```