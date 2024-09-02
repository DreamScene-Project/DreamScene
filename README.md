# DreamScene

This repository contains the official implementation for [DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling](https://arxiv.org/abs/2404.03575).

[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://dreamscene-project.github.io) [![arXiv](https://img.shields.io/badge/arXiv-2404.03575-b31b1b.svg)](https://arxiv.org/abs/2404.03575)

### News
- 2024-07-01: Our paper is accepted by ECCV2024 and to be published!

### TODO
- [x] Release the code of Formation Pattern Sampling (FPS) for single object.
- [ ] Release the code of entire Dreamscene for generating dream scenes and our demo video. (Before Sep. 14)

## Getting Start!
### Requirments

```bash
git clone https://github.com/DreamScene-Project/DreamScene.git
cd DreamScene

conda create -n dreamscene python=3.10
conda activate dreamscene

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu118/torch_stable.html

git clone --recursive https://github.com/DreamScene-Project/comp-diff-gaussian-rasterization.git
git clone https://github.com/YixunLiang/simple-knn.git

pip install comp-diff-gaussian-rasterization/
pip install simple-knn/

# Follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install point-e
git clone https://github.com/crockwell/Cap3D.git
cd Cap3D/text-to-3D/point-e/
pip install -e .
```

```sh
mkdir point_e_model_cache
# Optional: Initialize with better point-e
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_825kdata.pth
mv pointE_finetuned_with_825kdata.pth point_e_model_cache/
# Modify the parameter init_guided in the configuration file to pointe_825k

# or

wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_330kdata.pth
mv pointE_finetuned_with_330kdata.pth point_e_model_cache/
# Modify the parameter init_guided in the configuration file to pointe_330k
```

### Generate Single Object

```bash
python main.py --object --config configs/objects/sample.yaml
```

### Generate Entire Scenes

Coming Soon.

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [luciddreamer](https://github.com/EnVision-Research/LucidDreamer)
- [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)
- [threestudio](https://github.com/threestudio-project/threestudio)

## Citation
If you find it useful in your research, please consider citing our paper as follows:
```
@misc{li2024dreamscene3dgaussianbasedtextto3d,
  title={DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling}, 
  author={Haoran Li and Haolin Shi and Wenli Zhang and Wenjun Wu and Yong Liao and Lin Wang and Lik-hang Lee and Pengyuan Zhou},
  year={2024},
  eprint={2404.03575},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2404.03575}, 
}
```
