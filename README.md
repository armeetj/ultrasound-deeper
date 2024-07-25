# ultrasound-deeper

Reconstruction of lung images from ultrasound RF signal data. I designed and trained multiple architectures, ultimately increasing inference depth and improving on initial designs. I worked on this research project at the [Anima AI + Science Lab](http://tensorlab.cms.caltech.edu/users/anima/). I was mentored by [Dr. Jiayun Wang](http://pwang.pw/).

[proposal:pdf](https://github.com/user-attachments/files/16369650/surf_proposal_24.pdf)

## Task
The objective is to learn the mapping from radio frequency (RF) data to ultrasound images. The RF data has shape (time, num_transducers, num_receivers) (1509, 64, 128). At any given moment in time, this means 64 transducers emit a signal, and 128 transducers receive and record the signal. Each time sample RF[t, :, :] represents a "slice" of chest imaged. As time increases, we image _deeper_, past the chest wall and into the lung.

This video visualizes the RF data over time (1509 samples).

https://github.com/user-attachments/assets/724bbe23-09f8-482f-b496-4bc7f53f0e48

## Insight
Input data has simplified shape ~ (time, x, y).

2D U-Net architectures are trained on time slices with shape (x, y) so while they can learn spatial features, they fail to learn temporal features.

I implemented and trained a modified 3D U-Net to learn both spatial and temporal features, trained on the full temporal volume with shape (time, x, y) instead of just each temporal _slice_ with shape (x, y). As seen in results, this insight proved helpful.


## Models
- `nets.unet` contains a full implementation of both the 2D and 3D U-Nets
  - 4x 3D U-Net architectures (flagship)
  - 3x 2D U-Net architectures (2D slice training)
- `nets.cnn.VGG16` an initial baseline with an antiquated approach (mostly a sanity test)
- `nets.dense` various dense models (did not perform well)
- next steps: explore ViT & diffusion architectures


## Results

### Full Resolution
<img width="500" alt="image" src="https://github.com/user-attachments/assets/777fffdf-434f-4268-905d-b6848773c9e9">

From top to bottom (full resolution RF data):
- ground truth: 80 mm lung imaging depth
- 32 mm inference: existing models
- 48-**80 mm** inference: my trained 3D U-Net

### 16x Downsampling
<img width="500" alt="image" src="https://github.com/user-attachments/assets/5eab5582-0071-4387-aff3-7f962f706942">

<img width="500" alt="image" src="https://github.com/user-attachments/assets/a42ac953-f147-4ae5-902e-12a6e4b1d8bb">

Interestingly, my deepest model (bottom) was able to reconstruct some fine details where more shallow existing models fail.
As we increase the depth with which models are trained at, we are able to reconstruct more fine details, retaining accuracy.

Metrics and detailed results in paper.
