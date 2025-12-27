# Bank Marketing Data Preprocessing

This project focuses on data preprocessing using the Bank Marketing dataset.  
The objective is to clean, transform, and prepare the raw data for further analysis or machine learning tasks.

---

## 📌 Project Overview
Data preprocessing is a crucial step in any data science workflow.  
This project demonstrates a structured preprocessing pipeline including data inspection, cleaning, feature engineering, outlier handling, encoding, and scaling.

---

## 📊 Dataset
- **Name:** Bank Marketing Dataset  
- **Type:** Tabular data  
- **Domain:** Banking and Marketing  
- **Description:** Contains customer demographic details and marketing campaign information.

---

## ⚙️ Preprocessing Steps
- Loaded and inspected the dataset
- Displayed dataset shape, column names, and data types
- Explored data using indexing, slicing, `loc`, and `iloc`
- Checked and handled missing values
- Removed duplicate records
- Cleaned categorical variables
- Performed feature engineering (`age_group`)
- Treated outliers using the IQR method
- Encoded categorical variables
- Scaled numerical features using StandardScaler
- Saved the preprocessed dataset

---

## 📈 Outlier Handling
Outliers in numerical columns were treated using the **Interquartile Range (IQR)** method.  
Minimum and maximum values were printed before and after treatment to verify the impact.

---

## 📂 Output
- **Generated file:** `bank_marketing_preprocessed.csv`  
- The final dataset is clean, encoded, and scaled, ready for analysis or modeling.

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## ▶️ How to Run
1. Place the dataset file (`bank_marketing.csv`) in the project directory.
2. Run the Python script:

```bash
python data_preprocessing.py
