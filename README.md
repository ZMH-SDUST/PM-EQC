# Pose-Guided Multi-Cue Explicit Query Construction for Disambiguating Human-Object Interactions

:tada::tada::tada:The code and data are now open source!  
:rocket::rocket::rocket:We are organizing detailed tutorials on how to use the code to support community reproduction！

# Bridging Feature Misalignment and Semantic Confusion for Zero-Shot HOI Detection

## Dataset
Download the official HICO-DET and V-COCO datasets. The download files should organized as follows:

## Dependencies
1. Download [pocket](https://github.com/fredzzhang/pocket).
2. Download [DETR](https://github.com/fredzzhang/detr).
3. Download [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
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
3. Install the local package of [CLIP](https://github.com/openai/CLIP).
4. Download the CLIP weights [ViT-B-16.pt, ViT-L-14-336px.pt] to `VLA-Bridge/checkpoints/`.
5. Download the DETR weights [detr-r50-hicodet.pth, detr-r50-vcoco.pth] to `VLA-Bridge/checkpoints/`.
6. Download the pre-extracted features from [ADA-CM](https://github.com/ltttpku/ADA-CM).

## Train or Eval (HICO-DET)
### Train
```bash
python main_tip_ye.py --world-size 2--pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt
```
### Eval
```bash
`python main_tip_ye.py --world-size 2--pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --eval--resume ./checkpoints/hico/ckpt.pt
```
