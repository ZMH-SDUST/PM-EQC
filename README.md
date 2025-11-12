# Pose-Guided Multi-Cue Explicit Query Construction for Disambiguating Human-Object Interactions

:tada::tada::tada:The code and data are now open source!  
:rocket::rocket::rocket:We are organizing detailed tutorials on how to use the code to support community reproduction！

# Bridging Feature Misalignment and Semantic Confusion for Zero-Shot HOI Detection

## Dataset
Download the official [HICO-DET](https://huggingface.co/datasets/zhimeng/hico_det) and [V-COCO](https://github.com/s-gupta/v-coco) datasets.

## Dependencies
1. Download and install [pocket](https://github.com/fredzzhang/pocket).
2. Download and install [DETR](https://github.com/fredzzhang/detr).
3. Download and install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
4. Download [HICO-DET](https://github.com/fredzzhang/hicodet).
5. Download [V-COCO](https://github.com/fredzzhang/vcoco).
6. Download [PhysLab](https://github.com/ZMH-SDUST/PhysLab).
```
|- PM-EQC
|   |- pocket
|   |- DETR
|   |- OpenPose
|   |- hicodet  
|   |   |- hico_20160224_det  
|   |       |- annotations  
|   |       |- images  
|   |- vcoco  
|   |   |- mscoco2014  
|   |       |- train2014  
|   |       |- val2014      
```
7. Create Environment
```bash
conda create --name PMEQC python=3.8
conda activate PMEQC
pip install -r requriments.txt
```
8. Download pretrianed weights from [PViC](https://github.com/fredzzhang/pvic) and [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to `checkpoints/`.

## Train or Eval (HICO-DET, ResNet50-DETR + Openpose)
### Train
```bash
python main.py --pretrained checkpoints/detr-r50-hicodet.pth --output-dir outputs/pvic-detr-r50-hicodet
```
### Eval
```bash
python main.py --world-size 1 --batch-size 1 --eval --resume /path/to/model
```
