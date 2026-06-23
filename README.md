# 🛒 Online Retail Recommendation System

**CodTech IT Solutions — ML Internship**

| Field             | Details                              |
|-------------------|--------------------------------------|
| **Intern ID**     | CITS5099                             |
| **Full Name**     | Abhishek Prasad                      |
| **No. of Weeks**  | 4                                    |
| **Project Name**  | Online Retail Recommendation System  |
| **Project Scope** | Machine Learning                     |

---

## 📌 Project Overview

A recommendation system for an online retail store that suggests products to customers based on their purchase history and ratings. Three approaches are implemented and compared: Popularity-Based (cold start), Content-Based Filtering (TF-IDF), Collaborative Filtering (User-User cosine similarity), and a Hybrid blend of CF + Content-Based.

---

## 🗂️ Project Structure
online-retail-recommendation/

├── data/

│   ├── products.csv

│   ├── customers.csv

│   ├── transactions.csv

│   └── ratings.csv

├── outputs/

├── generate_data.py

├── visualize.py

├── model.py

├── main.py

└── requirements.txt
## ⚙️ How to Run
pip install -r requirements.txt

python main.py
## 🤖 Recommendation Approaches

| Approach | Method | Best For |
|---|---|---|
| **Popularity-Based** | Purchase count × avg rating | New users (cold start) |
| **Content-Based** | TF-IDF cosine similarity | Similar product discovery |
| **Collaborative Filtering** | User-User cosine similarity | Personalized recommendations |
| **Hybrid** | CF 60% + Content 40% | Best overall personalization |

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
