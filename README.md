# 📊 Financial Transaction Risk Classification

This project implements a Machine Learning system in Python for the analysis and classification of financial transactions. Its primary goal is to identify "high-risk" transactions using a **Logistic Regression** algorithm.

## 📁 Project Structure

The project is organized based on the separation of concerns principle to maintain clean and manageable code:

* `src/service.py`: The main entry point of the application. It orchestrates data loading, preprocessing, model training, and prediction extraction.
* `src/config.py`: Contains all system constants, file paths, and model hyperparameters.
* `src/data_loader.py`: Handles the safe ingestion of data from the `.csv` file and validates missing or incorrectly formatted values.
* `src/preprocessing.py`: Executes Feature Engineering (creating new variables, one-hot encoding, and logarithmic transformations).
* `src/model.py`: Implements a Scikit-Learn `Pipeline` (combining `StandardScaler` and `LogisticRegression` to prevent data leakage), trains the model, and saves it to the disk.
* `requirements.txt`: A list of all required external Python libraries.
* `data/`: The directory where the dataset (`finance_dataset.csv`) must be placed.
* `models/`: The directory where the trained model (`logistic_model.pkl`) is automatically saved.

## ⚙️ System Requirements

* **Python:** Version 3.10 or newer.
* A dataset named `finance_dataset.csv` placed inside the `data/` folder.

## 🚀 Installation and Execution

**Step 1: Install Dependencies** Open your terminal, navigate to the root folder of the project, and install the required packages by running:
```bash
pip install -r requirements.txt
