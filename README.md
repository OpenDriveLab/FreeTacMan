# FreeTacMan: Robot-free Visuo-Tactile Data Collection System

FreeTacMan is a robot-free, human-centric visuo-tactile
data collection system, featuring low-cost, high-resolution tactile sensors and a portable, cross-embodiment modular design. FreeTacMan transfers human visual perception, tactile sensing, and
motion control skills to robots efficiently by integrating visual and tactile data.

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
<!-- - [Demo](#demo)
  - [User Study](#user-study)
  - [Policy Rollouts](#policy-rollouts) -->
<!-- - [FreeTacMan's Performance](#freetacmans-performance)
  - [User Study](#user-study-1)
  - [Policy Rollouts](#policy-rollouts-1) -->
- [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
  - [Hardware Assembly](#hardware-assembly)
  - [Data Collection](#data-collection)
  - [Data Processing](#data-processing)
  - [Training](#training)
  - [Inference](#inference)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## ⭐ Overview

![FreeTacMan System Overview](figure/FreeTacMan_teaser.png)

We introduce FreeTacMan, a human-centric and robot-free data collection system for accurate and efficient robot manipulation. Concretely, we design a wearable data collection device with dual visuo-tactile grippers, which can be worn by human fingers for intuitive and natural control. A high-precision optical tracking system is introduced to capture end-effector poses while synchronizing visual and tactile feedback simultaneously. FreeTacMan achieves multiple improvements in data collection performance compared to prior works, and enables effective policy learning for contact-rich manipulation tasks with the help of the visuo-tactile information. 

📄 [Paper](https://arxiv.org/abs/XXXX.XXXXX) | 🚀 [Demo Page](https://freetacman.github.io) | 🛠️ [Hardware Guide](https://docs.google.com/document/d/1Hhi2stn_goXUHdYi7461w10AJbzQDC0fdYaSxMdMVXM/edit?addon_store&tab=t.0#heading=h.rl14j3i7oz0t)

✒️ Longyan Wu*, Checheng Yu*, Jieji Ren*, Li Chen, Ran Huang, Guoying Gu, Hongyang Li

📧 Primary Contact: Longyan Wu (im.longyanwu@gmail.com)

## 🦾 Highlights
- **Visuo-Tactile Hardware Sensor Design**: A portable, high-resolution, low-cost visuo-tactile hardware sensor designed for rapid adaptation across multiple robotic end-effectors. 
- **Tactile Data-collection System**: An in-situ, robot-free, real-time tactile data-collection system that leverages a handheld end effector and the proposed sensor to excel at diverse contact-rich tasks efficiently.
- **Policy Learning Enhanced by Tactile Pretraining**: Experimental validation shows that imitation policies trained with our visuo-tactile data achieve an average 50% higher success rate than vision-only approaches in a wide spectrum of contact-rich manipulation tasks.

<!-- ## 🎥 Demo

### User Study

 Fragile Cup | USB Plug | Texture Classification | Stamp Press | Calligraphy | Potato Chip | Tissue | Toothpaste |
|:-----------:|:--------:|:---------------------:|:-----------:|:-----------:|:-----------:|:------:|:----------:|
| <video src="video/user_study/FragileCupManipulation.mp4" width="200" controls></video> | <video src="video/user_study/USBPlugging.mp4" width="200" controls></video> | <video src="video/user_study/TextureClassification.mov" width="200" controls></video> | <video src="video/user_study/StampPressing.mp4" width="200" controls></video> | <video src="video/user_study/CalligraphyWriting.mov" width="200" controls></video> | <video src="video/user_study/PotatoChipGrasping.mp4" width="200" controls></video> | <video src="video/user_study/TissueGrasping.mp4" width="200" controls></video> | <video src="video/user_study/ToothpasteExtrusion.mp4" width="200" controls></video> |

### Policy Rollouts
(TODO: add video)

| Fragile Cup | USB Plug | Texture Classification | Stamp Press | Calligraphy |
|:-----------:|:--------:|:---------------------:|:-----------:|:-----------:|
| <video src="video/policy_rollouts/FragileCupManipulation.mov" width="200" controls></video> | <video src="video/policy_rollouts/USBPlugging.mov" width="200" controls></video> | <video src="video/policy_rollouts/TextureClassification.mp4" width="200" controls></video> | <video src="video/policy_rollouts/StampPressing.mov" width="200" controls></video> | <video src="video/policy_rollouts/CalligraphyWriting.mp4" width="200" controls></video> | -->

<!-- ## 🚀 FreeTacMan's Performance

### User Study
![Perfoemance of User Study](figure/userstudy.png)
*Figure 1: User study results comparing FreeTacMan with ALOHA and UMI across different metrics. FreeTacMan demonstrates superior performance in completion rate, collection efficiency, and CPUT score per task, while also excelling in user experience evaluation including control accuracy, ease of collection procedure, and stability.*

### Policy Rollouts
| Method | Fragile Cup | USB Plug | Texture Cls. | Stamp Press | Calligraphy | **Avg.** |
|:-------|:-----------:|:--------:|:------------:|:-----------:|:-----------:|:--------:|
| ACT (Vision-only) | 35 | 0 | 20 | 20 | 30 | **21** |
| Ours (+ Tactile w/o Pretraining) | 75 | 10 | 70 | 55 | 65 | **55** |
| Ours (+ Pretraining) | **80** | **20** | **90** | **85** | **80** | **71** |

*Table 3: Policy success rates (%) across contact-rich tasks. The visuo-tactile information, together with the pretraining strategy, greatly helps imitation learning for the contact-rich tasks.* -->

## 🎮 Getting Started

### Installation

#### Requirements

- Python 3.7+
- PyTorch 1.9+ (or compatible)
- CUDA 11.0+ (for GPU support)
- [Other dependencies](requirements.txt)

#### Clone the Repository

```bash
git clone https://github.com/yourusername/FreeTacMan.git
cd FreeTacMan
```

#### Install Dependencies

```bash
# Create a new conda environment (recommended)
conda create -n freetacman python=3.8
conda activate freetacman

# Install PyTorch (adjust version according to your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Hardware Assembly

For detailed hardware assembly instructions, please refer to our 🛠️ [Hardware Guide](https://docs.google.com/document/d/1Hhi2stn_goXUHdYi7461w10AJbzQDC0fdYaSxMdMVXM/edit?addon_store&tab=t.0#heading=h.rl14j3i7oz0t).

```bash
# Download 3D models
cd hardware/3d_models

# Print the parts using your 3D printer
```

### Data Collection

1. **Setup Environment**
   ```bash
   # Configure the tracking system
   python scripts/setup_tracking.py
   
   # Test the sensors
   python scripts/test_sensors.py
   ```

2. **Start Collection**
   ```bash
   # Start data collection
   python scripts/collect_data.py --task [task_name] --output [output_dir]
   ```

### Data Processing

1. **Preprocess Raw Data**
   ```bash
   # Process collected data
   python scripts/process_data.py --input [raw_data_dir] --output [processed_data_dir]
   ```

2. **Generate Dataset**
   ```bash
   # Create training dataset
   python scripts/create_dataset.py --input [processed_data_dir] --output [dataset_dir]
   ```

### Training

1. **Pretraining**
   ```bash
   # Start pretraining
   python scripts/train.py --config configs/pretrain.yaml
   ```

### Inference

1. **Load Model**
   ```bash
   # Load pretrained model
   python scripts/load_model.py --checkpoint [checkpoint_path]
   ```

2. **Run Inference**
   ```bash
   # Run inference on new data
   python scripts/inference.py --model [model_path] --input [input_data] --output [output_dir]
   ```

## 📝 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{wu2024freetacman,
  title={FreeTacMan: Robot-free Visuo-Tactile Data Collection System},
  author={Wu, Longyan and Yu, Checheng and Ren, Jieji and Chen, Li and Huang, Ran and Gu, Guoying and Li, Hongyang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🙏 Acknowledgements

We would like to thank:
- Huijie Wang for helping develop the demo page
- Zherui Qiu for helping organize and supervise the user study
- The faculty and students at Shanghai Jiao Tong University's Soft Robotics and Biodesign Lab for their support and assistance
- The authors of [ALOHA](https://github.com/tonyzhaozh/aloha) and [UMI](https://github.com/tonyzhaozh/umi) for their inspiring work
- All the participants in our user study for their valuable feedback
- The open-source community for their continuous support

## 📄 License

This project is licensed under the *** License - see the [LICENSE](LICENSE) file for details.
