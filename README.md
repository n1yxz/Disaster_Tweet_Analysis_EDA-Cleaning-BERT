# 🌪️ Disaster Tweet Analysis: EDA, Cleaning & BERT

A Natural Language Processing (NLP) project that classifies tweets as **disaster-related** or **not** using **exploratory data analysis**, **text preprocessing**, and a fine-tuned **BERT model**.

---

## 📑 Table of Contents

- [📌 Project Description](#-project-description)
- [✨ Key Features](#-key-features)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🙌 Credits](#-credits)
- [🪪 License](#-license)
- [📸 Screenshots & Links](#-screenshots--links)

---

## 📌 Project Description

This project analyzes tweets related to disasters using a full NLP pipeline — starting from **EDA** and **data cleaning** to training a **BERT-based classification model**. The objective is to **identify real disaster tweets** to aid in **faster emergency response** and **automated filtering** of social media content.

---

## ✨ Key Features

- 🔍 Interactive EDA on tweet metadata
- 🧹 Data cleaning & preprocessing pipeline
- 🤖 BERT-based binary classifier (Hugging Face Transformers)
- 📊 Evaluation using accuracy, F1-score, confusion matrix
- 📈 Visual insights on data and model performance

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/n1yxz/Disaster_Tweet_Analysis_EDA-Cleaning-BERT.git
   cd Disaster_Tweet_Analysis_EDA-Cleaning-BERT


1. **Clone the repository:**
   ```bash
   git clone https://github.com/n1yxz/Disaster_Tweet_Analysis_EDA-Cleaning-BERT.git
   cd Disaster_Tweet_Analysis_EDA-Cleaning-BERT
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

   Or, if using Jupyter, install dependencies in the notebook cells as needed.

---

## 🚀 Usage

- Open the Jupyter notebooks (`.ipynb`) in the repository.
- Follow the notebook cells step by step:
  - Perform EDA to understand the dataset.
  - Apply data cleaning and preprocessing.
  - Train and evaluate the BERT classifier.
- Modify parameters as desired for experimentation.

**Example:**
```bash
jupyter notebook Disaster_Tweet_Analysis_EDA_Cleaning_BERT.ipynb
```

---

## 📁 Project Structure

```
Disaster_Tweet_Analysis_EDA-Cleaning-BERT/
├── data/                # Dataset files (not included in repo)
├── notebooks/           # Main Jupyter notebooks
├── src/                 # Source code and scripts
├── outputs/             # Model outputs, predictions
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🙌 Credits

- **Contributors:** [n1yxz](https://github.com/n1yxz)
- **Datasets:** [Kaggle Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
- **Libraries:** HuggingFace Transformers, pandas, numpy, scikit-learn, matplotlib, seaborn

---

## 🪪 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📸 Screenshots & Links

![EDA Example](docs/eda_example.png)
![BERT Classification Report](docs/classification_report.png)

- [Project Repo](https://github.com/n1yxz/Disaster_Tweet_Analysis_EDA-Cleaning-BERT)
- [Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started)

---
