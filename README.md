
# IMDB Sentiment Analysis Using LSTM

A PyTorch-based implementation of sentiment analysis on the IMDB dataset using **LSTM networks**. This project includes a **Streamlit web app** for real-time sentiment predictions.

---

## Features

- **LSTM-based Model**: Captures long-term dependencies in textual data.
- **IMDB Dataset**: Utilizes the IMDB movie review dataset for training and evaluation.
- **Preprocessing Pipeline**: Incorporates TorchText utilities for tokenization and vocabulary building.
- **Interactive Web App**: Built with Streamlit for real-time sentiment analysis.
- **Model Checkpointing**: Saves the best model during training based on validation performance.

---

## Project Structure

```plaintext
.
├── train.py                # Training script for the LSTM model
├── main.py                 # Streamlit app for real-time sentiment prediction
├── embedding.ipynb         # Exploratory notebook for embedding analysis
├── requirements.txt        # Python dependencies
├── rnn_sentiment.pth       # Pretrained model weights
├── simple_rnn_imdb.h5      # Additional trained model weights
├── README.md               # Project documentation
└── .gitignore              # Specifies files to exclude from version control
```

---

## Requirements

- Python 3.8 or above
- PyTorch 1.10 or above
- Streamlit 1.10 or above
- TorchText
- Other dependencies are listed in `requirements.txt`.

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/vineethsaivs/imdb-sentiment-analysis-using-LSTM.git
cd imdb-sentiment-analysis-using-LSTM
```

### Step 2: Train the Model (Optional)

If you want to train the model from scratch:

```bash
python train.py
```

This will save the best model as `rnn_sentiment.pth`.

### Step 3: Run the Streamlit App

To launch the Streamlit app for sentiment analysis:

```bash
streamlit run main.py
```

---

## Technical Details

### 1. Dataset

The project uses the IMDB movie review dataset. Each review is labeled as `positive` or `negative`.

### 2. Model

The core of the project is an LSTM-based neural network implemented in PyTorch. Key model components:

- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Captures sequential information.
- **Fully Connected Layer**: Maps the output to a binary sentiment classification.

### 3. Training

The model is trained with:

- **Binary Cross Entropy Loss** for classification.
- **Adam Optimizer** for weight updates.
- **Validation Checkpointing** to save the best model.

### 4. Preprocessing

Text preprocessing includes:

- **Tokenization** using TorchText.
- **Truncation** to a fixed length of 500 tokens.
- **Conversion of tokens** to vocabulary indices.

---

## Example Workflow

- **Input**: "The movie was fantastic with brilliant performances!"
- **Model Output**: Positive sentiment with a prediction score of 0.85.
- **App Display**: Sentiment: Positive.

---

## Future Work

- Integrate additional datasets for improved generalization.
- Implement hyperparameter tuning for model optimization.
- Add deployment options for a hosted web app.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
