# Fine-Tuning LLaMA 3 on Medical Chatbot Dataset Using QLoRA

## Project Overview

This project focuses on fine-tuning **LLaMA 3 (8B parameters)** using **Quantized Low-Rank Adaptation (QLoRA)** to create an efficient and accurate medical chatbot. The chatbot is trained to respond to user queries with high reliability and context awareness, particularly in medical scenarios. By leveraging QLoRA and 4-bit quantization, the model achieves state-of-the-art fine-tuning efficiency with reduced memory requirements.

The model is trained on a dataset of patient-doctor conversations, where:
- The **user** provides a query (e.g., a patient describing symptoms).
- The **assistant** responds with advice (e.g., a doctor offering recommendations).

This setup ensures the model learns to generate structured, contextually relevant, and medically accurate responses.

---

## Key Highlights

### 1. **Efficient Fine-Tuning with QLoRA**
- **QLoRA** enables fine-tuning of large language models in low-precision (4-bit) environments while retaining performance.
- It uses **NF4 quantization** and **double quantization**, optimizing storage and compute resources.

### 2. **PEFT for Resource Optimization**
- The **PEFT (Parameter-Efficient Fine-Tuning)** framework focuses on specific layers, such as attention projection layers (`q_proj`, `k_proj`, `v_proj`, etc.), reducing the computational overhead.
- New low-rank matrices are added to these layers, enabling fast adaptation to the medical chatbot task.

### 3. **Medical Chatbot Dataset**
- The dataset comprises real-world styled patient-doctor conversations.
- Data formatting employs a structured **chat template** for training consistency, where inputs are tokenized into "user" and "assistant" roles.

### 4. **Seamless Integration with Hugging Face Hub**
- After fine-tuning, the adapter (fine-tuned layers) is saved and pushed to the Hugging Face Model Hub, enabling easy deployment and sharing.

---

## Training Approach

### **1. Model Initialization**
- The LLaMA 3 model is loaded with 4-bit quantization using the **BitsAndBytesConfig** framework.
- Quantized weights and activations allow the model to run efficiently on consumer-grade GPUs.

### **2. Dataset Preparation**
- The medical chatbot dataset is processed using a **chat template**, ensuring that the structure aligns with the user-assistant conversational flow.
- Example:
  ```json
  [
      {"role": "user", "content": "Hello doctor, I have bad acne. How do I get rid of it?"},
      {"role": "assistant", "content": "You should cleanse your skin twice daily and consider using products containing salicylic acid."}
  ]
### **3. Fine-Tuning**

Fine-tuning is performed using **SFTTrainer**, a specialized trainer designed for adapting large language models with **Parameter-Efficient Fine-Tuning (PEFT)**. The process involves optimizing only specific adapter layers, leaving the base model weights unchanged. This approach ensures efficient resource utilization while preserving the foundational knowledge of the pre-trained model.

Key aspects include:
- **Gradient Accumulation**:  
   To address memory constraints, gradient accumulation is employed. It enables the model to simulate larger batch sizes by aggregating gradients across smaller mini-batches before updating weights.
- **PEFT and LoRA**:  
   The model uses **Low-Rank Adaptation (LoRA)** to inject trainable weights into target modules. This technique focuses on fine-tuning critical layers like `q_proj`, `k_proj`, and `v_proj`, reducing the computational and storage overhead.

---

### **4. Evaluation**

During training, a separate test dataset is utilized to periodically evaluate the model's performance. This ensures:
- **Contextual Accuracy**: The model generates responses aligned with the input prompts.
- **Fluency**: The responses are coherent, grammatically correct, and contextually relevant.

The evaluation process is conducted at regular intervals, providing insights into the model's learning progression and highlighting any need for adjustments to hyperparameters or datasets.

---

### **5. Inference**

Once fine-tuned, the model is used for inference, generating responses based on user input formatted using a structured chat template. The steps include:
1. **Tokenizing Input**: User input is converted into tokens using the predefined tokenizer.
2. **Generating Output**: The fine-tuned model processes the tokens and generates a response.
3. **Decoding**: The tokenized output is decoded into human-readable text.

#### Example:
- **Input**:  
   `"I have a fever and a sore throat. What should I do?"`
- **Output**:  
   `"It sounds like you might have a viral infection. Drink plenty of fluids and rest. If the fever persists, consult a doctor."`

The model provides contextually accurate and actionable advice, demonstrating its adaptability to domain-specific tasks like medical consultations.

--- 

This structured fine-tuning and evaluation process ensures the final model performs effectively in real-world applications, particularly in healthcare chatbots where reliability and clarity are paramount.
