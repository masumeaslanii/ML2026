# Overfitting Mitigation Report: 2D CNN on CIFAR-100

In Session 6, a 2D CNN was trained on the CIFAR-100 dataset and suffered from overfitting. The goal of this exercise is to fix this problem using several different techniques and to compare their results, **without changing the network architecture**. All experiments were run on the free Google Colab GPU, so in addition to accuracy, the time cost of each method was also taken into account in the final selection.

The work was done in three stages:

1. **Individual experiments:** each technique tested on its own.
2. **Combined model** (`final.ipynb`): the most useful techniques combined.
3. **Final model** (`final_60epochs.ipynb`): the training recipe was refined (learning rate, schedule, augmentation), still with the same architecture.

## Baseline Model and Settings

- **Data:** The 50,000 training images were split with a fixed seed (42) into 45,000 train images and 5,000 validation images; the 10,000 test images are kept separate. Baseline augmentation: `RandomCrop` (padding=4) and `RandomHorizontalFlip`.

- **Architecture (fixed in all experiments):** Three convolutional blocks (Conv, BatchNorm, ReLU, MaxPool) with 64, 128, and 256 channels, followed by a fully connected layer with 128 neurons, Dropout with p=0.2, and a 100-class output layer.

- **Training:** SGD with lr=0.001 and momentum=0.9, CrossEntropy loss, batch size of 128, and 40 epochs.

**Observed problem:** In the baseline model, train accuracy climbed to 92.6% while validation accuracy stayed around 40% and even dropped after epoch 20 (validation loss rose from 92.2 to 104.8). This pattern is a clear sign of overfitting.

| Epoch | Train Acc | Val Acc | Val Loss |
| --- | --- | --- | --- |
| 10 | 35.94 | 37.38 | 98.7 |
| 20 | 55.23 | 40.30 | 92.2 |
| 30 | 75.81 | 36.06 | 103.5 |
| 40 | 92.63 | 39.16 | 104.8 |

## Stage 1: Individual Experiments

Each method was run in a separate notebook with the same baseline settings. In all of them, Early Stopping was enabled (patience=7 and min_delta=0.1 on validation accuracy, with the best weights saved):

- **Early Stopping:** on its own
- **Weight Decay:** with three values: 1e-4, 5e-4, 1e-3
- **Label Smoothing:** with a value of 0.1
- **LR Scheduler:** `CosineAnnealingLR`
- **Increased Dropout:** `Dropout2d` with p=0.1 after each convolutional block, and FC-layer Dropout raised to 0.5
- **ColorJitter:** with brightness, contrast, and saturation all set to 0.2
- **ColorJitter + RandomErasing:** `RandomErasing` added to ColorJitter with a probability of 0.25

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

- Early Stopping reduced training time from 921 to 577 seconds while preserving the best model. Because the validation set is small and each experiment was run only once, small differences (roughly 1 to 1.5 points or less) should be treated as noise.
- Weight Decay and Label Smoothing did not produce a significant improvement.
- The Scheduler with T_max=100 had almost no effect: training was much shorter than T_max, so the learning rate barely decreased.
- Increased Dropout clearly made performance worse (about an 8-point drop), and the gap between train and validation accuracy remained large.
- RandomErasing lowered accuracy compared to ColorJitter alone.
- ColorJitter was the only method that showed a modest improvement (about 1 point), but it roughly doubled the time cost.

## Stage 2: Combined Model (`final_final.ipynb`)

Based on the individual results, the following were combined:

- Early Stopping (patience=7)
- ColorJitter
- Weight Decay = 5e-4
- `CosineAnnealingLR` with T_max=40

Label Smoothing, RandomErasing, and extra Dropout were left out because they had a negative effect or no effect.

This model reached **44.82%** validation accuracy (epoch 38). Validation loss kept decreasing until the end of training and Early Stopping was never triggered, because with the gradual decrease of the learning rate the model continued to improve through the final epochs. However, the model was still training slowly (lr=0.001) and relied on a relatively weak regularizer, which motivated the next stage.

## Stage 3: Final Model (`final_60epochs.ipynb`)

The architecture was kept exactly the same. The following changes were made to the training recipe:

| Component | Stage 2 | Stage 3 (final) |
| --- | --- | --- |
| Optimizer | SGD, lr=0.001, momentum=0.9 | SGD, lr=0.005, momentum=0.9, Nesterov |
| Weight Decay | 5e-4 | 5e-4 |
| LR schedule | Cosine, T_max=40 | 3-epoch linear warmup (start factor 0.1), then Cosine with T_max=47 |
| Epochs | 40 | 50 |
| Augmentation | RandomCrop, Flip, ColorJitter | RandomCrop, Flip, **CutMix** (alpha=1.0, applied to each batch with probability 0.5) |
| Normalization | mean = std = 0.5 | mean and std computed from the CIFAR-100 training set |
| Early Stopping | patience=7 | patience=15 (never triggered) |

**Why these changes:**

- **Higher learning rate (0.005):** With lr=0.001 the model learned slowly. Preliminary short runs (not included in this repository) showed that a much higher value (0.05) was unstable, while moderate values worked well.
- **T_max equal to the training length:** The learning rate must actually decay to about zero by the end of training. When T_max was much longer than the run, or when Early Stopping cut the run before the decay phase, results were noticeably worse.
- **CutMix:** Training for more epochs without strong regularization mainly increased memorization (train accuracy rose far above validation accuracy). CutMix removed this behavior. ColorJitter was dropped, since it gave only a modest gain at about twice the training time.
- **Warmup and larger Early Stopping patience:** Warmup stabilizes the first epochs, and with a cosine schedule the validation accuracy can fluctuate for several epochs before improving, so a short patience would stop training too early.

**Results:** The final model reached **50.90%** validation accuracy at epoch 44, with a validation loss of 73.77. Training took 1106 seconds (about 22 s per epoch). Validation loss decreased almost continuously from about 174 to about 73, and validation accuracy stayed between 49.8% and 50.9% over the last 10 epochs, once the learning rate was close to zero.

**Test accuracy:** [add the test-set accuracy here]

## Comparison

| Metric | Baseline Model | Stage 2 (`final_final`) | Stage 3 (`final_60epochs`) |
| --- | --- | --- | --- |
| Best Val Acc | ~40.3% | 44.82% | **50.90%** |
| Best epoch | 20 | 38 | 44 |
| Train Acc at best epoch | 55.2% | 63.84% | 62.73%* |
| Val Loss at best epoch | 92.2 (lowest of the logged epochs) | 84.07 | 73.77 |
| Val Loss trend | Increasing after epoch 20 | Decreasing until the end | Decreasing until the end |

\* With CutMix, training accuracy is computed on mixed images against the original labels, so it is only an approximate indicator.

## Conclusion

With the architecture unchanged, the baseline's overfitting was effectively removed and validation accuracy rose from about 40% to **50.9%**. The most effective factors were a learning-rate schedule that decays fully within the training length (with a properly set T_max), a moderately higher learning rate with warmup, and CutMix augmentation. Individual regularization methods such as Weight Decay and Label Smoothing did not make a noticeable difference on their own, increased Dropout and RandomErasing hurt this small model, and ColorJitter gave only a modest gain at a high time cost. In the final model, validation loss decreased until the end, the train–validation gap is small, and accuracy plateaus once the learning rate reaches zero, which suggests the model is close to what this fixed architecture can achieve.

**Limitations:**

- Each experiment was run only once (a single seed). For greater confidence, each method can be repeated with several seeds.
- In Stage 3, several changes were made together (learning rate, warmup, CutMix, normalization, removal of ColorJitter), so the individual contribution of each change was not isolated.
