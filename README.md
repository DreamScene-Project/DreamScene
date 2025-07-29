# DreamScene
[Haoran Li](https://scholar.google.com/citations?user=Rxl8r70AAAAJ&hl=en), [Haolin Shi](https://i.4c43.work), [Wenli Zhang](https://github.com/kitty384/), [Wenjun Wu](https://github.com/saiyiii/), Yong Liao, [Lin Wang](https://vlislab22.github.io/vlislab/linwang.html), [Lik-hang Lee](https://www.lhlee.com/), [Pengyuan Zhou](https://github.com/pengyuan-zhou/),[Yuli Tian](https://github.com/lili174311/), [Kun Lan](https://github.com/klan-security),[Pan Hui](https://scholar.google.com/citations?user=dcDrhzMAAAAJ&hl=zh-CN),[Yuyang Wang](https://scholar.google.com/citations?user=D1HTbhEAAAAJ&hl=en),[Yonghui Wang](https://scholar.google.com.hk/citations?user=GGMWna4AAAAJ&hl=zh-CN), 

This repository contains the official implementation for [DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling](https://arxiv.org/abs/2404.03575).  
We extend it in [DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation](https://arxiv.org/abs/2507.13985), with a fully automatic pipeline.

**ECCV:** [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://dreamscene-project.github.io) [![arXiv](https://img.shields.io/badge/arXiv-2404.03575-b31b1b.svg)](https://arxiv.org/abs/2404.03575)

**Extension:**  [![Extension Page](https://img.shields.io/badge/Project-Webpage-blue)](https://jahnsonblack.github.io/DreamScene-Full/)  [![arXiv](https://img.shields.io/badge/arXiv-2507.13985-b31b1b.svg)](https://arxiv.org/abs/2507.13985)

Note: We compress these motion pictures for faster previewing.
 <table class="center">
    <tr style="line-height: 0">
      <td width=30% style="border: none; text-align: center">A DSLR photo of a modern living room. </td>
      <td width=30% style="border: none; text-align: center">An autumn park.</td>
      <td width=30% style="border: none; text-align: center">There are two astronauts on the moon, Sci-fi style</td>
    </tr>
    <tr style="line-height: 0">
      <td width=30% style="border: none"><img src="assets/livingroom.gif"></td>
      <td width=30% style="border: none"><img src="assets/autumn_park.gif"></td>
      <td width=30% style="border: none"><img src="assets/space.gif"></td>
    </tr>
 </table>

### News
- 2024-07-01: Our paper is accepted by ECCV2024 and to be published!
- 2025-07-29: We released the [webpage](https://jahnsonblack.github.io/DreamScene-Full/) for our ECCV extension!

### TODO
- [x] Release the code of Formation Pattern Sampling (FPS) for single object.
- [x] Release the code of entire DreamScene for generating dream scenes and our demo video.
- [ ] More samples, and tools for generating layout interactively.
- [ ] Release the code of the improved extension version with full pipeline.
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

If your device has more than 40G VRAM, you can run it with a single card. Otherwise, it is recommended to use dual cards.

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/scenes/sample_indoor.yaml

CUDA_VISIBLE_DEVICES=2,3 python main.py --config configs/scenes/sample_outdoor.yaml
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [luciddreamer](https://github.com/EnVision-Research/LucidDreamer)
- [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)
- [threestudio](https://github.com/threestudio-project/threestudio)

## Citation
If you find it useful in your research, please consider citing our papers DreamScene and its extension paper as follows:
```
@inproceedings{li2024dreamscene,
  title={Dreamscene: 3d gaussian-based text-to-3d scene generation via formation pattern sampling},
  author={Li, Haoran and Shi, Haolin and Zhang, Wenli and Wu, Wenjun and Liao, Yong and Wang, Lin and Lee, Lik-hang and Zhou, Peng Yuan},
  booktitle={European Conference on Computer Vision},
  pages={214--230},
  year={2024},
  organization={Springer}
}
```
```
@article{li2025dreamscene,
  title={DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation},
  author={Li, Haoran and Tian, Yuli and Lan, Kun and Liao, Yong and Wang, Lin and Hui, Pan and Zhou, Peng Yuan},
  journal={arXiv preprint arXiv:2507.13985},
  year={2025}
}
```