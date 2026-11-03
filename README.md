# Credit-card-fraud-detection
Installation and Setup

1. Clone the Repository

git clone https://github.com/tulsikerai09/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection

2. Create a Virtual Environment

python -m venv venv

3. Activate the Virtual Environment

Windows CMD:

venv\Scripts\activate

Windows PowerShell:

.\venv\Scripts\Activate.ps1

4. Install Dependencies

pip install -r requirements.txt

5. Download the Dataset

Download creditcard.csv from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded file inside:

data/creditcard.csv

6. Run the Streamlit Application

streamlit run app.py

The application will open in your browser, usually at:

http://localhost:8501
