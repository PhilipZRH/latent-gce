# Latent Gaussian Channel Empowerment
Implementation of our ICLR 2021 paper<br/>
<em>Efficient Empowerment Estimation for Unsupervised Stabilization</em><br/>
Ruihan Zhao, Kevin Lu, Pieter Abbeel, Stas Tiomkin<br/>
[\[OpenReview\]](https://openreview.net/forum?id=u2YNJPcQlwq)
[\[Project Page\]](https://sites.google.com/view/latent-gce)

## Getting Started

Create a conda environment, and install the necessary packages. Our codebase is developed and tested with TensorFlow version 1.15.0.

```
conda create -n latent-gce python=3.7
conda install tensorflow-gpu==1.15.0
pip install -r requirements.txt
pip install -e .
pip install -e custom_envs
```

For better visualization, the provided scripts automatically turn images into videos.
Please make sure ```ffmpeg``` is installed.

## Running Experiments

Sample scripts are included in the ```scripts``` folder.

### Sanity check with Ball-in-box environment

This script computes the empowerment of a point mass in a square box,
along with a control policy that drives the ball towards the center.
```
python3 scripts/ball_in_box_ppo.py
```

### State-based unsupervised pendulum swing-up

This script demonstrates how Latent-GCE swings up a pendulum when it is reset at the bottom,
all without any external reward signals.
```
python3 scripts/pendulum_state_ppo.py
```

### Pixel-based pendulum empowerment estimation

This script collects pixel observations from the pendulum environment,
and computes the empowerment value across the 2D state space.
```
python3 scripts/pendulum_pixels_empowerment.py
```

## Bibtex
```
@inproceedings{
zhao2021efficient,
title={Efficient Empowerment Estimation for Unsupervised Stabilization},
author={Ruihan Zhao and Kevin Lu and Pieter Abbeel and Stas Tiomkin},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=u2YNJPcQlwq}
}
```
