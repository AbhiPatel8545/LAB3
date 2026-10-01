# AIGC 5500 – Lab 03
## From Frozen Features to Fine-Tuning

### Project Overview

This project compares two ways of using a pretrained ResNet18 model for image classification on CIFAR-10:

1. **Feature extraction:** freeze the pretrained backbone and train only the final classification layer.
2. **Fine-tuning:** unfreeze the last backbone block (`layer4`) and train it together with the final classification layer (`fc`).

### Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset

- Dataset: CIFAR-10
- Number of classes: 10
- Training images used: 4,000
- Validation images used: 1,000
- Backbone: pretrained ResNet18 with ImageNet weights
- Random seed: 42
- Epochs per experiment: 5
- Batch size: 32

The 4,000 training images and 1,000 validation images are selected from the first 5,000 examples of the CIFAR-10 training set using a shuffled split with seed 42.

### Transform Pipelines

**Training transform**

- `RandomResizedCrop(224)`
- `RandomHorizontalFlip()`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

Random augmentation is used only for training images.

**Evaluation transform**

- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

The fixed evaluation transform is used for validation images.

### Feature Extraction

The pretrained ResNet18 backbone is frozen. Only the final fully connected classification layer is trainable. The frozen model is kept in `eval()` mode while the classification head is trained.

### Fine-Tuning

The fine-tuning experiment starts from pretrained ResNet18 ImageNet weights. The earlier backbone layers remain frozen. Only `layer4` and `fc` are trainable.

One Adam optimizer uses two parameter groups:

- `layer4` learning rate: `0.00001`
- `fc` learning rate: `0.001`

### Batch Normalization

During feature extraction, the frozen backbone is kept in `eval()` mode. If it were left in `train()` mode, its BatchNorm running mean and variance would continue to update using the new dataset. This would change the pretrained backbone's BatchNorm behaviour even though its weights are frozen.

### Actual Results

The following values are copied from the completed `LAB3.ipynb` notebook output.

| Stage | Validation Accuracy | Training Time | Trainable Parameters |
|---|---:|---:|---:|
| Feature Extraction | 78.8% | 859.61 seconds | 5,130 |
| Fine-Tuning | 75.4% | 1,178.18 seconds | 8,398,858 |

### Conclusion

In this experiment, feature extraction achieved 78.8% validation accuracy, while fine-tuning achieved 75.4%.

Feature extraction performed better in this particular run. The pretrained ResNet18 features were already useful for the CIFAR-10 classification task, so training only the new classification head produced the higher validation accuracy. Fine-tuning allowed the last backbone block (`layer4`) to update its weights, but this did not improve validation accuracy in this run.

This is a valid experimental result. Fine-tuning does not have to improve accuracy in every dataset or training setup.

### Known Limitations

- The experiment uses 5,000 images from the CIFAR-10 training set rather than the full dataset.
- Each approach is trained for 5 epochs.
- Training was performed on CPU in the recorded notebook run.
- Results may vary with hardware, software versions, data order, and training randomness.

