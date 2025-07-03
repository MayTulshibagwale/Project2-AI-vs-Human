# **Project 2: AI vs Human Text Detection Web Application**



#### 🚀 Project Overview



This project involves building a live web application using Streamlit that can analyze text from various documents (MS Word, PDF, plain text) and determine whether it was likely written by an AI or a human. The system provides probability scores indicating the likelihood of AI vs human authorship and offers a side-by-side model comparison.



#### ✨ Features

The web application provides the following functionalities:

* \*\*Text Input:\*\* Users can upload text files (.txt, .csv, .pdf, .docx) or directly type/paste text.
* \*\*Multiple Models:\*\* Supports classification using a suite of trained machine learning (SVM, Decision Tree, AdaBoost) and deep learning (CNN, LSTM, RNN) models.
* \*\*Instant Predictions:\*\* Real-time AI vs Human classification with confidence scores.
* \*\*Document Analysis:\*\* Ability to extract text from and analyze MS Word (.docx) and PDF (.pdf) documents.
* \*\*Visualizations:\*\* Provides insights through feature importance (in notebooks) and text statistics.
* ]\*\*Download Reports:\*\* Comprehensive analysis reports can be downloaded for batch processing.



#### 💻 Technical Stack



The project is built using:

* \*\*Main Framework:\*\* Streamlit with Python 
* \*\*Machine Learning:\*\* scikit-learn (for SVM, Decision Tree, AdaBoost) 
* \*\*Deep Learning:\*\* PyTorch (for CNN, LSTM, RNN) 
* \*\*Data Processing:\*\* pandas, numpy 
* \*\*Document Processing:\*\* PyPDF2, pdfplumber, python-docx 
* \*\*Text Preprocessing:\*\* NLTK, spaCy 
* \*\*Embeddings:\*\* Word2vec, GloVe, fastText Embeddings (specifically GloVe used in this project) 
* \*\*Visualization:\*\* matplotlib, seaborn, plotly, wordcloud 



\## 📁 Project Structure

The project repository is organized as follows:



Project2/

&nbsp;	

├── data/                   

│   ├── AI\_vs\_human\_train\_dataset.xlsx 

|── Mrunmayee\_Tulshibagwale\_project2\_R11815197.ipynb 

├── models/                 

│   ├── adaboost\_model.pkl      # Trained AdaBoost model

│   ├── CNN.pkl                 # Trained CNN model (PyTorch state\_dict)

│   ├── decision\_tree\_model.pkl # Trained Decision Tree model

│   ├── LSTM.pkl                # Trained LSTM model (PyTorch state\_dict)

│   ├── MAX\_SEQUENCE\_LENGTH.pkl # Maximum sequence length used for DL models

│   ├── RNN.pkl                 # Trained RNN model (PyTorch state\_dict)

│   ├── svm\_model.pkl           # Trained SVM model

│   ├── tfidf\_vectorizer.pkl    # Fitted TF-IDF Vectorizer (for ML models)

│   └── word\_to\_idx.pkl         # Dictionary mapping words to indices (for DL models)

├── app.py  

|-- requirements.txt

└── README.md                      





#### ⚙️ Setup and Installation



To set up and run this project locally, follow these steps:



1\.  **\*\*Clone the Repository:\*\***

&nbsp;   ```bash

&nbsp;   git clone <your-repository-url>

&nbsp;   cd ai\_human\_detection\_project

&nbsp;   ```



2\.  **\*\*Create a Virtual Environment (Recommended):\*\***

&nbsp;   ```bash

&nbsp;   python -m venv venv

&nbsp;   # On Windows:

&nbsp;   .\\venv\\Scripts\\activate

&nbsp;   # On macOS/Linux:

&nbsp;   source venv/bin/activate

&nbsp;   ```



3\.  **\*\*Install Dependencies:\*\***

&nbsp;   Install all required Python packages. Create a `requirements.txt` file in your project's root directory with the following content (including versions if you prefer strict reproducibility, otherwise just the names):

&nbsp;   ```

&nbsp;   streamlit

&nbsp;   pandas

&nbsp;   numpy

&nbsp;   joblib

&nbsp;   scikit-learn

&nbsp;   torch

&nbsp;   nltk

&nbsp;   spacy

&nbsp;   PyPDF2

&nbsp;   python-docx

&nbsp;   pdfplumber

&nbsp;   matplotlib

&nbsp;   seaborn

&nbsp;   plotly

&nbsp;   wordcloud

&nbsp;   openpyxl

&nbsp;   ```

&nbsp;   Then run:

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



4\.  **\*\*Download NLTK Data and spaCy Model:\*\***

&nbsp;   These are required for text preprocessing.

&nbsp;   ```bash

&nbsp;   python -m nltk.downloader stopwords wordnet punkt

&nbsp;   python -m spacy download en\_core\_web\_sm

&nbsp;   ```



5\.  **\*\*Place Training Data:\*\***

&nbsp;   Ensure your `AI\_vs\_human\_train\_dataset.xlsx` file is placed in the `data/` directory.



6\.  **\*\*Generate Trained Models and Resources:\*\***

&nbsp;   The trained models (`.pkl`, `.pt`) and Deep Learning resources (`.pkl`, `.pt`) \*\*are not included in the repository by default\*\* due to their size. You must generate them by running the training notebook:

&nbsp;   \* Open `notebooks/Mrunmayee\_Tulshibagwale\_project2\_R11815197.ipynb` in Jupyter or VS Code.

&nbsp;   \* \*\*Run all cells from top to bottom.\*\* This process will:

&nbsp;       \* Load and preprocess the `AI\_vs\_human\_train\_dataset.xlsx`.

&nbsp;       \* Generate TF-IDF features.

&nbsp;       \* Load GloVe embeddings and create vocabulary/embedding matrix for DL models.

&nbsp;       \* Prepare PyTorch DataLoaders.

&nbsp;       \* Train and save all Machine Learning models (`svm\_model.pkl`, `decision\_tree\_model.pkl`, `adaboost\_model.pkl`) to the `models/` directory.

&nbsp;       \* Train and save all Deep Learning models (`CNN.pkl`, `LSTM.pkl`, `RNN.pkl`) to the `models/` directory.

&nbsp;       \* Save DL resources (`word\_to\_idx.pkl`, `MAX\_SEQUENCE\_LENGTH.pkl`, `embedding\_matrix.pt`) to the `models/` directory.

&nbsp;   \* \*\*Verify\*\* that the `models/` directory now contains all the necessary `.pkl` and `.pt` files.



#### ▶️ How to Run the Application

Once the setup is complete and all models/data are in place, you can run the Streamlit web application:



```bash

streamlit run app.py

This command will open the application in your default web browser.



**📈 Development and Analysis**

The notebooks/Mrunmayee\_Tulshibagwale\_project2\_R11815197.ipynb notebook contains all the code for data loading, preprocessing, feature engineering, model development (training and hyperparameter tuning for ML and DL models), and performance evaluation (metrics, confusion matrices, ROC curves). You can open this notebook with Jupyter or VS Code to explore the development process in detail.



**📄 Code Documentation**

Throughout your codebase, you should include thorough code documentation and comments that explain complex algorithms, design decisions, and any non-obvious implementation details. This includes within 



app.py, your Jupyter notebook, and any other .py files you create.



**📹 Demo Video \& Presentation**

A concise demo video (3-5 minutes) that showcases your application's key features, walks through the user interface, demonstrates predictions on sample texts, and highlights the unique aspects of the AI detection system will be submitted separately.



Author: Mrunmayee Tulshibagwale 

R11815197    

