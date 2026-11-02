\# An Explainable Type-Aware Hybrid AI Framework for Data Quality Monitoring and Error Detection



\## Project Overview



This project presents an AI-based Data Quality Monitoring framework designed to identify and explain data quality issues in structured datasets.



The framework combines type-aware feature engineering, Random Forest classification, supporting anomaly-detection signals and duplicate-related signals to identify different categories of data quality errors.



It also provides human-readable explanations and an interactive Streamlit dashboard to help users understand detected issues and monitor data quality.



\## Key Features



\- \*\*Type-Aware Error Detection:\*\* Identifies different data quality problems using features designed for different data types.

\- \*\*Hybrid AI Approach:\*\* Combines Random Forest predictions with supporting anomaly-detection and duplicate-related signals.

\- \*\*Multiple Error Categories:\*\* Detects missing values, invalid values, outliers, inconsistencies and duplicates.

\- \*\*Explainable Results:\*\* Generates human-readable explanations for flagged records.

\- \*\*Interactive Dashboard:\*\* Displays performance metrics, error distributions, feature importance and error explanations.

\- \*\*Cross-Dataset Evaluation:\*\* Evaluates the framework using additional datasets to examine its performance across different data structures.



\## Error Categories



The framework focuses on five major data quality issues:



1\. Missing Values

2\. Invalid Values

3\. Outliers

4\. Inconsistencies

5\. Duplicate Records



\## Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Random Forest

\- Isolation Forest

\- Local Outlier Factor (LOF)

\- One-Class SVM

\- Streamlit

\- Matplotlib and Seaborn



\## Dataset Information



The primary experimental dataset is Online Retail II.



Additional datasets used for cross-dataset evaluation:



\- Bank Marketing

\- Telco Customer Churn



Controlled synthetic corruption was introduced to create known error labels for experimental evaluation.



The datasets are not included in this repository. Obtain them from their respective sources and place them in the appropriate project directories before running the experiments.



\## Primary Experimental Results – V5.7



The following results were obtained on the primary Online Retail II test set.



| Metric | Result |

|---|---:|

| Accuracy | 98.90% |

| Precision | 90.29% |

| Recall | 91.34% |

| F1-Score | 90.81% |



\### Error-Type F1-Scores



| Error Category | F1-Score |

|---|---:|

| Missing Value | 74.70% |

| Invalid Value | 99.24% |

| Outlier | 95.06% |

| Inconsistency | 100.00% |

| Duplicate | 87.90% |



\## Cross-Dataset Evaluation



The framework was also evaluated using two additional datasets.



| Dataset | Accuracy | Precision | Recall | F1-Score |

|---|---:|---:|---:|---:|

| Bank Marketing | 98.54% | 77.45% | 99.56% | 87.12% |

| Telco Customer Churn | 97.68% | 74.67% | 80.00% | 77.24% |



These experiments use separately configured evaluation pipelines and controlled synthetic corruption. They do not establish performance on naturally occurring errors.



\## Project Structure



```text

AI-Data-Quality-Monitoring/

│

├── datasets/

│   ├── raw/

│   ├── corrupted/

│   └── processed/

│

├── models/

│

├── notebooks/

│

├── paper/

│

├── results/

│   ├── metrics/

│   │   └── v5\_7/

│   └── cross\_dataset\_v2\_3/

│

├── src/

│   ├── data\_loader.py

│   ├── dq\_rule\_engine.py

│   ├── error\_injection.py

│   ├── error\_explanation.py

│   ├── model\_scoring\_v2.py

│   ├── type\_aware\_hybrid\_v5\_7.py

│   └── dashboard\_and\_figures.py

│

├── requirements.txt

└── README.md

```



\*Note: The repository may contain additional source files and supporting modules.\*



\## Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/manogna006/AI-Data-Quality-Monitoring.git

cd AI-Data-Quality-Monitoring

```



\### 2. Create a Virtual Environment



Windows PowerShell:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Run the Streamlit Dashboard



```bash

streamlit run .\\src\\dashboard\_and\_figures.py

```



The dashboard will open in your browser, usually at:



```text

http://localhost:8501

```



\## Important Experimental Limitations



\- Ground-truth labels were generated through controlled synthetic corruption.

\- Some corruption-related features remain in the V5.7 feature set and may contribute to optimistic results for certain error categories.

\- The duplicate-related signal uses dataset-level information and therefore has a transductive component.

\- Row-level splitting may allow related records to appear across different partitions.

\- Cross-dataset results vary by dataset and error category.

\- Further leakage-controlled experiments, real-world validation, component ablations and repeated evaluations are needed.



The reported results should therefore be interpreted within the experimental setup described above.



\## Future Enhancements



\- Evaluation using naturally occurring data quality errors.

\- Improved contextual inconsistency detection.

\- More robust duplicate identification.

\- Leakage-controlled evaluation and group-based data splitting.

\- Component-wise comparisons and ablation studies.

\- Confidence-score calibration.

\- Scalability evaluation on larger datasets.



\## Author



\*\*Manogna Chereddy\*\*



\## Repository



\[AI-Data-Quality-Monitoring – GitHub](https://github.com/manogna006/AI-Data-Quality-Monitoring)



\---



\*This project was developed as an academic capstone project focused on explainable AI-based data quality monitoring and error detection.\*

