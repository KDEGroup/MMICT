# MMICT: **Boosting Multi-Modal Fine-Tuning with In-Context Examples**

Source code for TOMM 2024 paper "MMICT: Boosting Multi-Modal Fine-Tuning with In-Context Examples" [[arXiv preprint](https://arxiv.org/abs/2312.06363)].

## Environment

The required environment is included in `requirements.txt`.

## Dataset Preparation
We train and test our model on:
+ [MSVD](https://www.cs.utexas.edu/~ml/clamp/videoDescription/)

+ [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

+ [COCO](https://cocodataset.org/#download)

+ [VQAv2](https://visualqa.org/download.html)

## How to run

To train the model:

```python
bash run.sh
```

## Acknowledgments
We thank the developers of [LAVIS](https://github.com/salesforce/LAVIS), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), for their public code release.