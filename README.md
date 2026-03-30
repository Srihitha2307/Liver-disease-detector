# 🔬 Liver Disease Detector (BYOP Capstone)

An interactive Machine Learning web application designed to predict liver disease risk and provide transparent explanations for its decisions using SHAP.

## 🚀 Project Overview
This project addresses the challenge of early liver disease diagnosis. It utilizes a **Random Forest Classifier** trained on the Indian Liver Patient Dataset. To ensure clinical utility, the app provides a "Decision Analysis" chart for every prediction, showing which lab markers most influenced the result.

## 📂 Repository Structure
*Click on any file or folder to view its contents:*

- [**`app.py`**](./app.py): The main Streamlit web application.
- [**`train.py`**](./train.py): Script to preprocess data and train the saved models.
- [**`data/`**](./data/): Contains the [**`indian_liver_patient.csv`**](./data/indian_liver_patient.csv) dataset.
- [**`models/`**](./models/): Stores serialized assets ([`liver_model.pkl`](./models/liver_model.pkl), [`scaler.pkl`](./models/scaler.pkl), [`encoder.pkl`](./models/encoder.pkl)).
- [**`Project_Report.pdf`**](): The detailed technical documentation for VITyarthi submission.
- [**`requirements.txt`**](./requirements.txt): List of necessary Python libraries.

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-github-username>/<your-repo-name>.git
   cd Liver_Disease_Detector
