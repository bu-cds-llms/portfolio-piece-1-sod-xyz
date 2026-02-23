# Evaluating Sentiment Model Robustness Across Genre Subdomains

## Overview

Machine learning models often perform well on the data they were trained on, yet struggle when applied to slightly different data. This phenomenon, known as domain shift, occurs when a machine learning model's training data (source domain) differs statistically from the data it encounters during deployment (target domain), leading to performance degradation and is common in real-world applications. A sentiment model trained on product reviews, for example, may not perform equally well on social media posts or news articles.



This project examines that problem in a controlled setting. Using the IMDB movie review dataset that was completed during our DS593 Lab 1 assignment, I investigate whether a sentiment classifier trained on general movie reviews performs equally well on specific genre-based subsets such as horror, comedy, romance, and action.



Because the IMDB dataset does not include explicit genre labels, I construct genre-like subsets using keyword-based weak supervision. The goal is not only to measure accuracy, but to understand how changes in language distribution affect model performance.


## Central Research Question:

Does a sentiment model trained on general movie reviews maintain its performance when applied to genre-specific subsets, or does domain shift reduce classification accuracy?



\## Dataset:

This project uses the IMDB Movie Review Dataset from Stanford which can be found \[!\[here](https://ai.stanford.edu/~amaas/data/sentiment/)



Full Dataset:

* 50,000 total labeled reviews
* 25,000 labeled training reviews
* 25,000 labeled test reviews
* Evenly split between positive and negative sentiment



Dataset Preparation for This Project: 

To allow efficient experimentation while maintaining statistical balance, I use a subset of:

* 5,000 total movie reviews
* 2,500 positive reviews
* 2,500 negative reviews



The data is randomly shuffled and split into:

* 80% Training set
* 20% Test set



Download and Extraction

The notebook automatically:

* Downloads the dataset from Stanford’s website (if not already present);
* Extracts it to the locally; and 
* Loads the selected subset into a pandas DataFrame.



\## Methodology 

\### 1. Constructing Genre-Based Subsets (Weak Supervision)



The IMDB dataset does not include genre labels. To simulate genre subdomains, I use keyword-based rules to assign reviews to genre-like categories.



Examples:

\- Horror: `gore`, `monster`, `scary`, `slasher`

\- Romance: `love story`, `relationship`, `chemistry`

\- Comedy: `funny`, `hilarious`, `jokes`

\- Action: `fight`, `explosion`, `battle`



This approach allows analysis without manual annotation. However, it introduces noise because keyword presence does not perfectly represent true genre depiction. This benefits as it reflects real-world scenarios where labeled data is incomplete or imperfect.



This method also:

* Enables domain-specific analysis
* Introduces controlled label noise
* Simulates distant supervision strategies used in applied NLP





\### 2. Feature Representation: TF-IDF



Reviews are transformed into TF-IDF vectors. TF-IDF was selected because:

* It reduces the impact of very frequent words that carry little meaning
* It emphasizes words that are distinctive within documents
* It produces sparse, interpretable representations
* It is a strong and well-understood baseline for text classification



Vocabulary and inverse document frequency (IDF) values are computed using training data only. This prevents data leakage and ensures a fair evaluation of results.



\### 3. Model: Multinomial Naive Bayes



A Multinomial Naive Bayes classifier is used. This model is appropriate because:

* It performs well on sparse, high-dimensional text features
* It is computationally efficient compared to other models
* It requires minimal hyperparameter tuning
* It allows inspection of feature log-probabilities for interpretability



The goal of this project is not to optimize performance with complex models, but to examine robustness under domain variation using a clear and interpretable baseline.



\### 4. Experimental Design



Three evaluation settings are used:

1. General → General

* Train and test on randomly split IMDB data.



2\. Subset → Subset (In-Domain)

* Train and test within each genre subset.



3\. General → Subset (Cross-Domain Transfer)

* Train on general IMDB data and test on genre-specific subsets.



This design isolates the effect of domain shift. If performance decreases in the third setting compared to the first, it suggests that genre-specific language patterns affect model generalization.

## 5. Key Results

* **Baseline Performance:**  
  The General → General model achieved strong accuracy (~0.88–0.90), establishing a stable benchmark under no domain shift.

* **In-Domain Variation Across Genres:**  
  Subset → Subset performance varied by genre. Comedy achieved the highest in-domain accuracy (~0.91), while romance was substantially lower (~0.64), suggesting that some genres are inherently harder for sentiment classification.

* **No Performance Degradation Under Cross-Domain Transfer:**  
  Contrary to expectations, the General → Subset model consistently outperformed the Subset → Subset models across all genres. This indicates that training on a larger and more diverse dataset improved robustness rather than harming it.

* **Directional Error Pattern in Horror Reviews:**  
  Misclassification analysis showed that positive horror reviews were often predicted as negative. Words like “gore,” “gross,” and “devil” were interpreted negatively by the general model, even when used positively within the horror context. This provides qualitative evidence of how genre-specific language can shift sentiment interpretation.

* **Overall Conclusion:**  
  In this controlled setting, data diversity and training size outweighed the effects of moderate domain variation. The results suggest that broader training data may enhance robustness rather than degrade performance under mild domain shifts.




\## Limitations



Several limitations should be considered when interpreting the results.



* \*\*Weak Genre Labels:\*\* Genre subsets are constructed using keyword rules rather than true metadata. This introduces noise: some reviews may be misclassified, and others may belong to multiple genres. As a result, the subdomains are approximations rather than precise genre categories.
* \*\*Model Simplicity:\*\* Multinomial Naive Bayes assumes independence between words and relies on a bag-of-words representation. It does not capture context, word order, or negation. Some performance degradation may therefore reflect representational limits rather than domain shift alone.
* \*\*Restricted Evaluation Scope:\*\* The analysis focuses primarily on accuracy. While appropriate for balanced data, it does not fully describe class-specific errors or model calibration under distribution shift.
* \*\*Controlled Domain Shift:\*\* All subsets are drawn from the same underlying IMDB corpus. Real-world cross-domain transfer (e.g., across platforms or writing styles) would likely produce stronger distributional differences.



Overall, the findings illustrate how lexical variation influences sentiment performance, but broader generalization claims require more diverse datasets and modeling approaches.



\## Future Direction

Several extensions would strengthen this analysis.



1. Comparing Naive Bayes with a linear model such as Logistic Regression would clarify whether domain sensitivity arises from the classifier or from the TF-IDF representation itself.



2\. Replacing keyword-based genre assignment with true genre metadata, or allowing multi-label genres, would reduce label noise and sharpen domain boundaries.



3\. Finally, evaluating transfer across entirely different review types (e.g., movie to book reviews) would test whether the observed effects generalize beyond controlled subdomains within IMDB, especially considering many fans have strong opinions positively for one review type such as books to compared to another review types such as movies.


\## How to Run:

1\. Install required packages (see below)

2\. Open \*notebook/Portfolio\_Piece\_1.ipynb\*

3\. Run all cells from top to bottom

4\. Visualizations will be saved in the \*outputs/\* directory

(Note: The first run may take several minutes to download dataset)



\## Requirements:

\- Python 3.9+

\- numpy

\- pandas

\- matplotlib

\- scikit-learn


\## Repository Structure:

├── README.md

├── requirements.txt

├── notebooks/

│   └── portfolio_1.ipynb
│   └── lab_1.ipynb

├── src/              # downloaded dataset

└── outputs/          # figures, tables, etc.



