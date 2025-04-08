# USING TRADITIONAL AND DEEP MACHINE LEARNING TO PREDICT EMERGENCY ROOM TRIAGE LEVELS

## 📄 Abstract

Accurate triage in emergency rooms is crucial for efficient patient care and resource allocation. This project explores methods to predict triage levels using both traditional machine learning techniques (Logistic Regression, Random Forest, XGBoost) and deep learning models (CNNs, LSTMs, Attention-based architectures).

The models were trained and evaluated using structured and unstructured data from emergency department visits at a local Turkish hospital. A key challenge addressed was processing Turkish-language documents and incorporating features specific to the Turkish medical system.

We applied text embedding techniques including Bag of Words (BoW), Word2Vec, and BERT to patient complaints. The best performing model was XGBoost combined with Word2Vec embeddings, achieving:
- **AUC:** 86.7%
- **Accuracy:** 81.5%
- **Weighted F1 Score:** 68.7%

Our findings highlight that integrating patient history and using effective embedding strategies significantly improve triage level prediction.

## 🧠 Models & Techniques Used

- **Traditional Machine Learning:** Logistic Regression, Random Forest, XGBoost  
- **Deep Learning:** CNN, LSTM, Attention Mechanisms  
- **Embedding Methods:** Bag of Words (BoW), Word2Vec, BERT (via Hugging Face Transformers)

## 🏗️ Project Structure
<pre>
Predicting-Emergency-Room-Triage-Levels/
├── <b>etl/</b>
│   ├── data_creation.py              (masked)
│   ├── feature_engineering.py        (masked)
│   ├── feature_preprocessing.py      (masked)
│   ├── preprocessing_vars.ipynb
├── <b>experiments/</b>
│   ├── results.ipynb
│   ├── pickles/                      (saved models)
├── <b>exploratory_analysis/</b>
│   ├── plots.ipynb
│   ├── tsne_model_for_chief_complaints.ipynb
├── <b>models/</b>
│   ├── ml_model.ipynb
│   ├── text_embedding_methods.ipynb
</pre>


> ⚠️ **Note**: Some files are intentionally masked due to patient data privacy.

## 🔧 Requirements

- Python 3.8+
- Jupyter
- scikit-learn
- xgboost
- gensim
- transformers
- pytorch
- pandas, numpy, matplotlib, seaborn

## 📊 Results & Findings

See experiments/results.ipynb for model performance comparison and evaluation metrics.

## 📁 Data

Due to privacy concerns, the dataset used in this project is not publicly shared. However, the code and structure demonstrate how to preprocess and model similar healthcare text data.

## 📎 Citation

If you use this codebase in your work or wish to cite the project, please use the following BibTeX entry:

@misc{triageprediction2025,
  author       = {Mehmet Yıldırım},
  title        = {Using Traditional and Deep Machine Learning to Predict Emergency Room Triage Levels},
  year         = {2025},
  howpublished = {\url{https://github.com/ericantona7/Predicting-Emergency-Room-Triage-Levels}},
  note         = {Accessed: 2025-04-08}
}

## 🤝 Contact

For collaboration or questions, feel free to reach out via email: m.yildirim961*@gmail.com

### ✅ How to Add to Your Paper as a Reference

In your paper (if you're using LaTeX), you can cite it like this:
```latex
\bibitem{triageprediction2025}
Mehmet Yıldırım. \textit{Using Traditional and Deep Machine Learning to Predict Emergency Room Triage Levels}. GitHub Repository. Available at: \url{https://github.com/ericantona7/Predicting-Emergency-Room-Triage-Levels}, Accessed April 8, 2025.
```
If you’re using plain text, in the references section, write:
```
Mehmet Yıldırım. Using Traditional and Deep Machine Learning to Predict Emergency Room Triage Levels. GitHub Repository. https://github.com/ericantona7/Predicting-Emergency-Room-Triage-Levels. Accessed April 8, 2025.
```
