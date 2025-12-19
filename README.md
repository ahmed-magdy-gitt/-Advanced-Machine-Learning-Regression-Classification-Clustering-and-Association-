# 🚀 Comprehensive Machine Learning Portfolio: From Scratch Implementation to Advanced Algorithms

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLxtend](https://img.shields.io/badge/MLxtend-Library-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📋 Executive Summary
This repository features a complete implementation of core Machine Learning algorithms, ranging from **Supervised** to **Unsupervised** learning techniques. The project highlights a deep technical understanding by comparing **Manual Implementations (Math-based)** against industry-standard libraries, and analyzing the performance trade-offs between different algorithms.

---

## 📂 Project Modules & Technical Details

### 1️⃣ Linear Regression: Manual Math vs. Scikit-Learn
* **Goal:** Predict student writing scores based on reading scores.
* **Implementation Details:**
    * **Manual Implementation:** Built a Linear Regression model from scratch (Pure Python) using the **Least Squares Method** to calculate Slope ($m$) and Intercept ($b$).
    * **Automated Implementation:** Used `sklearn.linear_model.LinearRegression` for validation.
* **Key Result:** Proved that the manual mathematical approach yields identical predictions to the optimized library, demonstrating a solid grasp of the underlying algorithms.

### 2️⃣ Classification Comparison: KNN vs. Decision Tree
* **Goal:** Classify data points into categorical labels with high accuracy.
* **Key Techniques:**
    * **Preprocessing:** Applied `StandardScaler` (Crucial for KNN distance calculation) and `OneHotEncoder`.
    * **Model Tuning:** Compared **K-Nearest Neighbors (Distance-based)** against **Decision Trees (Rule-based)**.
* **Outcome:** Analyzed performance metrics (Accuracy, F1-Score) to determine the best model for the dataset.

### 3️⃣ Customer Segmentation: K-Means vs. BIRCH
* **Goal:** Group mall customers based on behavior using Unsupervised Learning.
* **Algorithms:**
    * **K-Means:** Standard centroid-based clustering.
    * **BIRCH:** Hierarchical clustering designed for large datasets.
* **Evaluation:** Used **Silhouette Score** and **Elbow Method** to optimize the number of clusters. The comparison highlighted the efficiency of BIRCH in handling noise versus the simplicity of K-Means.

### 4️⃣ Market Basket Analysis: Apriori vs. FP-Growth
* **Goal:** Discover strong association rules and product correlations in transactional data.
* **Comparison:**
    * **Apriori Algorithm:** Traditional level-wise search.
    * **FP-Growth (Frequent Pattern Growth):** Efficient tree-based structure (No candidate generation).
* **Insight:** Conducted a performance benchmark showing **FP-Growth's superior speed** and memory efficiency over Apriori on larger datasets.

---

## 🛠️ Technologies & Libraries
| Domain | Tools Used |
| :--- | :--- |
| **Data Processing** | `Pandas`, `NumPy` |
| **Visualization** | `Matplotlib`, `Seaborn` |
| **Machine Learning** | `Scikit-Learn`, `MLxtend` |
| **Techniques** | `Regression`, `Classification`, `Clustering`, `Association Rules` |

## 🚀 How to Run
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/ML-Algorithms-Portfolio.git](https://github.com/YourUsername/ML-Algorithms-Portfolio.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn mlxtend
    ```
3.  **Explore the Notebooks:**
    * Run `Final_Project.ipynb` for the Regression analysis.
    * Run `Classification.ipynb` for model comparison.
    * Run `Association_Role.ipynb` for Market Basket Analysis.

---
*Created by [Your Name] | 2025*
