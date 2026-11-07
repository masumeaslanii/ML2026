# Overfitting Mitigation Report: 2D CNN on CIFAR-100

**Masume Aslani** — 40452111001

In Session 6, a 2D CNN was trained on the CIFAR-100 dataset and suffered from overfitting. The goal of this exercise is to fix this problem using several different techniques and to compare their results. All experiments were run on the free Google Colab GPU, so in addition to accuracy, the time cost of each method was also taken into account in the final selection.

## Baseline Model and Settings

- **Data:** The 50,000 training images were split with a fixed seed (42) into 45,000 train images and 5,000 validation images; the 10,000 test images are kept separate. Baseline augmentation: `RandomCrop` (padding=4) and `RandomHorizontalFlip`.

- **Architecture:** Three convolutional blocks (Conv, BatchNorm, ReLU, MaxPool) with 64, 128, and 256 channels, followed by a fully connected layer with 128 neurons, Dropout with p=0.2, and a 100-class output layer.

- **Training:** SGD with lr=0.001 and momentum=0.9, CrossEntropy loss, batch size of 128, and 40 epochs.

**Observed problem:** In the baseline model, train accuracy climbed to 92.6% while validation accuracy stayed around 40% and even dropped after epoch 20 (validation loss rose from 92.2 to 104.8). This pattern is a clear sign of overfitting.

| Epoch | Train Acc | Val Acc | Val Loss |
| --- | --- | --- | --- |
| 10 | 35.94 | 37.38 | 98.7 |
| 20 | 55.23 | 40.30 | 92.2 |
| 30 | 75.81 | 36.06 | 103.5 |
| 40 | 92.63 | 39.16 | 104.8 |

## Tested Methods

Each method was run in a separate notebook with the same baseline settings. In all of them, Early Stopping was enabled (patience=7 and min_delta=0.1 on validation accuracy, with the best weights saved):

- **Early Stopping:** on its own
- **Weight Decay:** with three values: 1e-4, 5e-4, 1e-3
- **Label Smoothing:** with a value of 0.1
- **LR Scheduler:** `CosineAnnealingLR`
- **Increased Dropout:** `Dropout2d` with p=0.1 after each convolutional block, and FC-layer Dropout raised to 0.5
- **ColorJitter:** with brightness, contrast, and saturation all set to 0.2
- **ColorJitter + RandomErasing:** `RandomErasing` added to ColorJitter with a probability of 0.25

## Results of Individual Experiments

| Method | Best Val Acc (%) | Best Epoch | Time (s) |
| --- | --- | --- | --- |
| Early Stopping | 40.84 | 18 | 577 |
| Weight Decay 1e-4 | 40.36 | 17 | 590 |
| Weight Decay 5e-4 | 40.76 | 18 | 575 |
| Weight Decay 1e-3 | 40.64 | 18 | 584 |
| Label Smoothing | 40.36 | 19 | 605 |
| Scheduler (T_max=100) | 40.90 | 20 | 644 |
| More Dropout | 32.12 | 25 | 738 |
| ColorJitter | **41.84** | 25 | 1204 |
| ColorJitter + RandomErasing | 39.62 | 18 | 1005 |

**Analysis:**

- Early Stopping reduced training time from 921 to 577 seconds while preserving the best model. Because the validation set is small and each experiment was run only once, small differences should be treated as noise.
- Weight Decay and Label Smoothing did not produce a significant improvement.
- The Scheduler with T_max=100 had almost no effect: training was much shorter than T_max, so the learning rate barely decreased.
- Increased Dropout clearly made performance worse (about an 8-point drop), and the gap between train and validation accuracy remained large.
- RandomErasing lowered accuracy compared to ColorJitter alone.
- ColorJitter was the only method that showed a modest improvement, but it roughly doubled the time cost.

## Final Model

Based on the results, the final model combines the following:

- Early Stopping (patience=7)
- ColorJitter
- Weight Decay = 5e-4
- `CosineAnnealingLR` with T_max=40

Label Smoothing, RandomErasing, and extra Dropout were left out of the final model because they had a negative effect or no effect.

| Metric | Baseline Model | Final Model |
| --- | --- | --- |
| Best Val Acc | ~40.3% | 44.82% (epoch 38) |
| Train Acc at best epoch | 55.2% (epoch 20) | 63.84% |
| Val Loss | 92.2 (minimum) → 104.8 (end) | 84.03 (at end of training) |
| Val Loss trend | Increasing after epoch 20 | Decreasing until the end of training |

In the final model, validation loss kept decreasing until the end of training and Early Stopping was never triggered, because with the gradual decrease of the learning rate the model continued to improve through the final epochs. The gap between train and validation accuracy also shrank from about 53 points to about 19 points.

## Conclusion

The baseline model's overfitting was effectively reduced by combining ColorJitter data augmentation, gradual learning-rate decay with `CosineAnnealingLR`, Weight Decay, and Early Stopping: validation accuracy rose from about 40% to 44.8%, the validation loss curve no longer trended upward, and the train–validation gap narrowed. Contrary to expectations, increased Dropout and RandomErasing hurt this small model. Another key finding is that individual regularization methods (Weight Decay and Label Smoothing) did not make a noticeable difference on their own, and reducing the learning rate, with a properly set T_max, was the most effective factor.

**Limitation:** Each experiment was run only once (a single seed). For greater confidence, each method can be repeated with several seeds.
