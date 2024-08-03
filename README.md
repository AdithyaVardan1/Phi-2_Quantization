# Quantization and Fine-Tuning of SLMs for Edge Deployment

This project explores techniques for efficiently running Speech Language Models (SLMs) on resource-constrained edge devices, such as the Microsoft Phi-2 model. The focus is on achieving both high accuracy and low computational footprint through quantization and fine-tuning.

## Techniques Investigated:

* *QLORA:* This quantization method leverages low-rank approximations to represent weights in SLMs with minimal accuracy loss.
* *Gradient Low-Rank Projection:* This technique reduces the dimensionality of gradients during training, enabling efficient fine-tuning of pre-trained SLMs on edge devices.

## Project Goals:

* Develop a framework for quantizing and fine-tuning SLMs for deployment on edge devices.
* Evaluate the effectiveness of QLORA and gradient low-rank projection in reducing model size and computational complexity while maintaining accuracy.
* Optimize SLMs for real-time inference on resource-limited devices like the Microsoft Phi-2.

## Potential Benefits:

* Enables deployment of powerful SLMs on edge devices, facilitating applications like real-time voice assistants and local language processing.
* Reduces reliance on cloud infrastructure for SLM inference, improving privacy and reducing latency.
* Contributes to the development of more efficient and portable SLMs for broader real-world applications.
