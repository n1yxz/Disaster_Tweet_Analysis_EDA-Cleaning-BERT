# Disaster_Tweet_Analysis_EDA-Cleaning-BERT

## Table of Contents
- [Project Description](#project-description)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)
- [Screenshots & Links](#screenshots--links)

---

## Project Description

This project analyzes disaster-related tweets using exploratory data analysis (EDA), data cleaning, and a BERT-based classification model. The primary goal is to distinguish between tweets that refer to real disasters and those that do not, aiding in rapid disaster response and information filtering.

---

## Key Features

- Comprehensive EDA on disaster tweet datasets
- Data cleaning and preprocessing pipelines
- Implementation of a BERT-based classifier for disaster tweet detection
- Evaluation metrics (accuracy, F1, confusion matrix)
- Visualizations for data insights and model performance

---

## Installation

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

## Usage

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

## Project Structure

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

## Credits

- **Contributors:** [n1yxz](https://github.com/n1yxz)
- **Datasets:** [Kaggle Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
- **Libraries:** HuggingFace Transformers, pandas, numpy, scikit-learn, matplotlib, seaborn

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Screenshots & Links

![EDA Example](docs/eda_example.png)
![BERT Classification Report](docs/classification_report.png)

- [Project Repo](https://github.com/n1yxz/Disaster_Tweet_Analysis_EDA-Cleaning-BERT)
- [Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started)

---

> _Replace any placeholder text or images with your actual project content for a personalized README!_
