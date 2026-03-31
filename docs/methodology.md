# Methodology

## 1. Problem Formulation

This work studies privacy-preserving representation learning for facial images. The
objective is to learn a latent representation that remains useful for a designated
downstream task while suppressing information associated with sensitive biometric
attributes. In the current implementation, the downstream utility task is binary
smiling classification, whereas the privacy-sensitive attributes are facial identity
and gender.

Let `X` denote an input face image, `y_u` the utility label, and `y_p` the private
attribute labels. The model first maps `X` to an intermediate feature sequence and then
transforms that sequence through a privacy filter to produce a latent representation
`Z`. A utility classifier predicts `y_u` from `Z`, while an adversary attempts to
recover `y_p` from the same representation. Training follows the standard adversarial
representation-learning objective:

`min_{theta, phi} max_{psi} L_u(T_phi(F_theta(X)), y_u) - lambda L_p(A_psi(F_theta(X)), y_p)`

where `F_theta` is the privacy filter, `T_phi` is the utility model, `A_psi` is the
privacy adversary, and `lambda` controls the privacy-utility tradeoff. Intuitively,
the model is encouraged to preserve task-relevant information while removing features
that make the private attributes predictable.

## 2. Dataset and Task Definition

The current experiments are conducted on CelebA, a large-scale face-attribute dataset
containing aligned celebrity face images, binary facial attributes, and identity
annotations. The dataset is downloaded through torchvision and converted into the
repository's metadata-driven image format by
[prepare_celeba.py](/home/carl/sujosh/APPR-PHOTOS/scripts/prepare_celeba.py).

The current task definition is as follows:

- Utility task: `smiling` versus `not_smiling`
- Privacy targets: `speaker_id` (identity) and `gender`

Each row in `data/raw/celeba/metadata.csv` includes:

- `filename`: relative image path
- `utility_label`: smiling or not_smiling
- `speaker_id`: CelebA identity label
- `gender`: derived from the CelebA `Male` attribute
- `age`: currently left blank

Under this formulation, the repository treats CelebA as a generic privacy-preserving
image-learning benchmark in which a socially meaningful facial attribute serves as the
utility objective and identity-related information serves as the privacy objective.

## 3. Data Preparation and Splitting

Images are loaded by the dataset implementation in
[image_dataset.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/data/image_dataset.py). Each
image is converted to RGB, normalized to the `[0, 1]` range, and resized to
`224 x 224`. Labels are read from `metadata.csv`; utility labels are mapped to class
indices and speaker identifiers are mapped to a contiguous identity vocabulary.

The current training pipeline uses a speaker-stratified split implemented in
[utils.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/data/utils.py). Speakers are first
partitioned, and all samples belonging to a given speaker are assigned to the same
split. This prevents identity leakage across partitions. The split ratios are:

- Training: `70%`
- Validation: `15%`
- Test: `15%`

This split policy is a defensible choice for preventing direct identity overlap across
partitions. However, it has methodological implications for identity privacy
evaluation: because validation and test speakers are disjoint from the training
speakers, a speaker-classification adversary evaluated on held-out speakers is not
measuring the same phenomenon as a closed-set identity classifier. Accordingly,
identity privacy metrics in the current setup should be interpreted with caution.

## 4. Model Architecture

The model is composed of four functional components:

1. an image feature extractor,
2. a privacy filter,
3. a utility classifier, and
4. a multi-head privacy adversary.

### 4.1 Image Feature Extractor

The image encoder is defined in
[image_cnn.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/features/image_cnn.py). It is a
compact convolutional neural network consisting of repeated
`Conv2d -> BatchNorm2d -> ReLU` stages with stride-2 downsampling. The final feature map
is flattened across spatial locations to produce a tensor of shape `(B, D, T)`, where
`B` is the batch size, `D` the feature dimension, and `T` the number of flattened
spatial tokens.

For the current CelebA experiments:

- Input channels: `3`
- Hidden channel sequence: `[32, 64, 96]`
- Output feature dimension: `128`

This representation design allows the downstream privacy filter and task model to
operate over a sequence-like structure rather than a single pooled vector.

### 4.2 Privacy Filter

The privacy filter is implemented in
[privacy_filter.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/models/privacy_filter.py).
Its role is to transform the image feature sequence into a representation that retains
utility while weakening privacy leakage.

The filter contains:

- a multi-scale temporal convolution block with kernel sizes `3`, `7`, and `15`,
- additional `Conv1d` layers with `InstanceNorm1d`, `ReLU`, and dropout,
- an optional Variational Information Bottleneck (VIB).

The use of multiple kernel sizes allows the filter to model dependencies at different
spatial-token scales. Instance normalization is used within the filter to reduce the
reliance on batch-level statistics. When enabled, the VIB module parameterizes a
Gaussian latent distribution and contributes a KL regularization term to the loss.

In the current configuration:

- Hidden dimension: `256`
- Output dimension: `128`
- Number of filter layers: `3`
- VIB: enabled
- `vib_beta = 0.005`
- Dropout: `0.1`

### 4.3 Utility Classifier

The utility classifier is defined in
[task_model.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/models/task_model.py). It first
applies learnable attention pooling over the latent sequence and then predicts the
utility label through a small multilayer perceptron. For the current task, the output
space is binary:

- `not_smiling`
- `smiling`

This architecture allows the model to adaptively emphasize different latent positions
when making the utility decision.

### 4.4 Privacy Adversary

The privacy adversary is implemented in
[adversary.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/models/adversary.py). It is a
multi-head classifier with a shared trunk and one classification head per private
attribute. The adversary is attached to the latent representation through a Gradient
Reversal Layer (GRL), implemented in
[gradient_reversal.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/models/gradient_reversal.py).

The GRL leaves the forward pass unchanged but multiplies the backward pass by `-lambda`.
Consequently:

- the adversary itself is trained to improve private-attribute prediction, while
- the feature extractor and privacy filter receive reversed gradients, pushing them
  toward privacy-invariant representations.

The current adversary uses:

- a shared trunk dimension of `128`,
- a `gender` head with 2 classes,
- a `speaker_id` head with 10177 classes,
- dropout of `0.3`.

The adversary uses mean pooling rather than attention pooling over the latent sequence,
which is an intentional stability choice in the current implementation.

## 5. Optimization and Training Procedure

Training is orchestrated by
[trainer.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/training/trainer.py), with the
current GPU experiment defined in
[celeba_nvidia.yaml](/home/carl/sujosh/APPR-PHOTOS/configs/experiment/celeba_nvidia.yaml)
and
[celeba_baseline.yaml](/home/carl/sujosh/APPR-PHOTOS/configs/experiment/celeba_baseline.yaml).

Two optimizers are used:

- a main optimizer for the feature extractor, privacy filter, and utility model,
- an adversary optimizer for the privacy classifier.

The current training hyperparameters are:

- Optimizer: Adam
- Main learning rate: `1e-3`
- Adversary learning rate: `5e-4`
- Batch size: `128` on GPU
- Epochs: `30`
- Gradient clipping: `1.0`
- Teacher distillation: disabled in the active CelebA configuration

### 5.1 Joint Objective

The combined training loss is implemented in
[losses.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/training/losses.py). In the current
configuration, the total objective is:

`L_total = L_utility + L_KL + lambda L_privacy`

where:

- `L_utility` is cross-entropy for smiling classification,
- `L_KL` is the VIB regularization term,
- `L_privacy` is the average privacy classification loss across all valid privacy heads.

Because the adversary is attached through the Gradient Reversal Layer, the same privacy
loss improves the adversary while simultaneously pushing the upstream representation
toward reduced privacy leakage.

Missing privacy labels are handled by masking invalid entries (`label < 0`) so that
losses are computed only for valid targets.

### 5.2 Privacy Scheduling

The privacy pressure is not applied at full strength from the first epoch. Instead, the
trainer uses a DANN-style sigmoid schedule to increase the effective GRL strength over
time. The target privacy coefficient in the current CelebA run is:

- `lambda_privacy = 0.1`

This gradual schedule stabilizes early learning by allowing the utility classifier to
form a reasonable decision boundary before privacy pressure becomes dominant.

### 5.3 Adversary Refresh

The current implementation periodically refreshes the adversary to reduce overfitting to
a single static attack model. Every `20` epochs, the adversary is reinitialized and
trained alone for `5` epochs. During these retraining epochs, the main model is not
updated. This procedure is intended to maintain a meaningful privacy challenge over the
course of training.

### 5.4 Checkpoint Selection

Checkpoints are saved after every epoch, and the best checkpoint is selected using
validation utility UAR only. In other words, the current repository optimizes privacy
during training but chooses the final checkpoint according to utility preservation
rather than an explicit privacy-utility selection rule.

## 6. Evaluation Protocol

Evaluation is implemented in
[evaluator.py](/home/carl/sujosh/APPR-PHOTOS/src/aapr/evaluation/evaluator.py). The
evaluation pipeline reports both utility and privacy metrics on the held-out split.

### 6.1 Utility Metrics

The current utility evaluation includes:

- Unweighted Average Recall (UAR)
- Weighted Accuracy (WA)
- Macro F1

These metrics quantify how well the learned representation preserves the smiling task.
Among them, UAR is particularly useful when class balance is not perfect, whereas WA
provides the familiar overall accuracy measure.

### 6.2 Privacy Metrics

The current privacy evaluation includes:

- Gender accuracy
- Gender UAR
- Speaker identity accuracy
- Speaker identity UAR
- A nearest-centroid mutual-information proxy `MI(Z; speaker)`

Gender-based privacy metrics are comparatively straightforward to interpret in the
current setup. Identity-based privacy metrics are more subtle because the split policy
is speaker-disjoint; a very low identity accuracy on the validation or test set does not
automatically imply strong de-identification, since the adversary is evaluated on
identities absent from training.

### 6.3 Implementation Note on Standalone Evaluation

The current codebase now saves and reloads the trained feature extractor as part of the
checkpoint state. This is important because standalone evaluation requires the same
feature extractor weights used during training. Earlier evaluation artifacts generated
before this fix may understate test utility and should therefore be treated as stale.

## 7. Current Experimental State

The current active experiment is the CelebA GPU run stored under
`outputs/celeba_nvidia`. At present, the best saved checkpoint on disk corresponds to
epoch `2`, with validation metrics stored inside the checkpoint of approximately:

- `utility_uar = 0.9230`
- `utility_wa = 0.9230`
- `utility_f1 = 0.9229`

These results indicate that the current pipeline is capable of preserving strong utility
performance on the smiling task under adversarial training. At the same time, the
privacy interpretation remains asymmetric:

- gender privacy metrics can be used as a reasonable proxy for attribute suppression,
- identity privacy metrics should be treated as provisional because of the current
  speaker-disjoint evaluation design.

Accordingly, the present methodology is well suited for reporting utility preservation
and initial privacy-pressure behavior, while any strong closed-set identity privacy
claim would require a more specialized CelebA evaluation protocol than the one
currently implemented.
