# FedCDT-LLM: An Intelligent Scoring Framework for Digital Clock Drawing Test (dCDT) Based on Federated Learning and Local Large Language Models
## Introduction：
This project is the open-source implementation of the FedCDT-LLM framework, designed to address key challenges in early screening for neurodegenerative diseases (such as dementia) using the Digital Clock Drawing Test (dCDT) data: privacy protection, cross-institutional Non-Independent and Identically Distributed (Non-IID) data, and insufficient unimodal information.

FedCDT-LLM innovatively combines Multimodal Fusion with Centralized Federated Learning (FedAvg). It leverages a local Large Language Model (LLM, such as GPT-5) to generate high-quality medical semantic text descriptions, which are then dynamically fused with dCDT image features. This significantly enhances the dCDT scoring model's accuracy, generalization ability, and clinical interpretability.
## Core Technical Features
1.Federated Learning (FedAvg): A centralized federated learning framework built on FedLab, ensuring that the original patient data is not exposed during collaborative training across multiple clients.

2.Multimodal Enhancement: Integrates ResNet-18 (for image features) and a frozen BERT (for text features) to effectively combine the structural information from dCDT images with the medical semantic information from text.

3.Dynamic Attentive Fusion: Employs the Attentive Fusion mechanism to dynamically learn the weights of the image and LLM-generated text modalities, enabling adaptive feature fusion.

4.Text Robustness Analysis: Provides separate centralized experimental code for an in-depth analysis of the model's dependence on and robustness to variations in text data quality (random masking/shuffling).
## Project Structure and Module Division
The project code is divided into two main parts: the Main Experiment Modules (implementing the core FedCDT-LLM workflow using FedLab) and the Centralized Robustness Experiment Module (used for the discussion section of the paper).
### Part I: FedCDT-LLM Core Framework (Main Experiment)
<img width="643" height="280" alt="1760682218690" src="https://github.com/user-attachments/assets/8c916c19-3d4f-4142-862c-ec8d4ed4e5b8" />

### Part II: Centralized Text Modality Effect Experiment (Robustness Analysis)
<img width="642" height="119" alt="1760682275599" src="https://github.com/user-attachments/assets/c66ec52e-a9f9-4401-b672-04b486a77e00" />
