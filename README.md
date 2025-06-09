# Phi3-EdgeQuant-Agent

**Phi3-EdgeQuant-Agent** is an advanced project focused on deploying a **quantized Phi-3-mini (3.8B) LLM agent** on edge devices. Built with high efficiency and research-grade modularity, it leverages quantization techniques to enable large language models to run on low-resource hardware such as laptops or edge accelerators.

> Target Audience: Researchers, AI Engineers, ML Edge Developers, and Open-Source Contributors

## 🔍 Objective

To build a **self-quantizing Phi3 LLM agent** that can:

* Operate efficiently on edge devices (e.g., RTX 3050, Jetson, RPi 5 with accelerator).
* Preserve inference capabilities close to full precision.
* Serve as a modular base for further development (e.g., AutoGen agents, RAG systems, embedded NLP tools).

## 🚀 Features

* ✅ Loads and quantizes Phi-3-mini (3.8B) locally
* ✅ Structured scripts for inference, quantization, and model loading
* ✅ Edge-optimized with 8-bit quantization support
* ✅ Clear directory and modular Python scripts
* ✅ Clean integration with future AutoGen or RAG-based agents

## 🏛 Architecture

```
Phi3-EdgeQuant-Agent/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_agent.py               # Entry point for agent execution
├── src/
│   ├── model_loader/
│   │   ├── download_phi3_model.py   # Downloads Phi-3-mini to local path
│   │   └── inference_phi3_local.py # Handles local inference
│   └── quantization/
│       └── quantize_phi3_8bit.py   # Quantizes model to 8-bit
```

## 📚 Setup

### 1. Clone Repo

```bash
git clone https://github.com/pyschofives/Phi3-EdgeQuant-Agent.git
cd Phi3-EdgeQuant-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model

```bash
python src/model_loader/download_phi3_model.py
```

> Downloads Phi-3-mini to `models/` directory

### 4. Quantize the Model

```bash
python src/quantization/quantize_phi3_8bit.py
```

### 5. Run Inference

```bash
python scripts/run_agent.py
```

## 🔢 Model Used

* `microsoft/Phi-3-mini-4k-instruct`
* Quantized using `bitsandbytes`, `transformers`, and `AutoGPTQ` (optional extension)

## 🌍 Applications

* Offline LLM agents on laptops and embedded systems
* R\&D in LLM compression and deployment
* Integration with lightweight assistant UIs

## ✨ Coming Soon

* ✅ AutoGen agent wrapper
* ✅ Performance benchmarks vs FP16
* ✅ Dynamic quantization + on-device chat GUI
* ✅ Paper-ready experiments + Colab notebook

## 🚩 Contributions

PRs are welcome. For major changes, open an issue first to discuss what you would like to change.

## 📅 License

MIT License

---

Maintained by **CudaBit Team**
Lead Developer: [Stifler (AI Engineer)](https://github.com/STiFLeR7)

---

**If you like this repo, please give it a star!** ✨
