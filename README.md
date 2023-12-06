# Conditional Variational Autoencoder Data Augmentation for Robustness to Instance-Dependent Noise

## Research Paper
For detailed information about our study and findings, please refer to our research paper. You can access it here: [Report Paper](report_paper.pdf)

## Abstract
This paper addresses the critical challenge in deep learning of obtaining high-quality, accurately labeled data. In real-world scenarios, label noise is often unavoidable and can significantly hinder model performance. Traditional artificially generated label noise does not adequately represent real-world conditions. We propose a novel approach using instance-dependent noise (IDN) as a benchmark for modeling real-world label noise, coupled with a deep generative method for effective handling of this issue.

## Introduction
The effectiveness of deep learning systems heavily relies on the quality of the labeled data. However, accurate and noise-free labels are challenging and time-consuming to obtain. Our research focuses on addressing the gap between artificially generated label noise and real-world label noise, using IDN as a more representative benchmark.

## Methodology
We introduce the Conditional Variational Autoencoder (CVAE) as a novel method for data augmentation on desired class labels. This approach aims to enhance the robustness of deep learning models against the detrimental effects of label noise.

### Conditional Variational Autoencoder (CVAE)
- The CVAE is designed to encode generalized class-conditional features.
- It helps in mitigating the influence of noisy labels by enhancing data quality through augmentation.

## Dataset and Evaluation
- **Dataset Used**: MNIST dataset.
- **Evaluation Method**: Comparative analysis with several state-of-the-art techniques under less noisy conditions.

## Results
- The experimental results demonstrate that our CVAE-based method outperforms existing techniques in scenarios with lower levels of noise.
- The CVAE-based data augmentation shows robustness to small percentages of noise, underscoring its potential in real-world image classification scenarios.

## Conclusion
The study confirms the potential of the CVAE in improving the performance of image classification models, particularly in real-world settings where label noise is a common issue. Our findings highlight the importance of developing techniques that are tailored to the nuances of real-world data.
