# Tunisian Dialect Abusive Language Detection

This project focuses on detecting abusive language in comments written in the Tunisian dialect. Using a combination of web scraping, deep learning, and state-of-the-art NLP models, the system identifies abusive content in Tunisian comments from YouTube.

---

## 🚀 Features

- **Web Scraping**: Scraped Tunisian comments from YouTube using the YouTube Data API.
- **Deep Learning**: Leveraged CNN-LSTM architecture for abusive language classification.
- **Transformer Models**: Used pre-trained BERT models fine-tuned for the Tunisian dialect.
- **Tunisian Dialect Focus**: Tailored for comments in Tunisian Arabic dialect.

---

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Znullptr/Tunisian-Dialect-Abusive-Language-Detection.git
   cd Tunisian-Dialect-Abusive-Language-Detection
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API key for YouTube Data API in the scrapper script:
   ```bash
   YOUTUBE_API_KEY=your_api_key
   ```
## 🗂️ Dataset

- Comments were scraped from YouTube using the YouTube Data API.
- The dataset was preprocessed to clean text, remove noise, and label abusive and non-abusive comments.
- Stored in the `Dataset/` folder.

---

## 💻 Models Used

### CNN-LSTM
- Combines the power of **Convolutional Neural Networks (CNNs)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for sequential modeling.
- Implemented in **TensorFlow/Keras**.

### BERT (Bidirectional Encoder Representations from Transformers)
- Fine-tuned on the preprocessed Tunisian comments dataset.
- Enabled state-of-the-art performance in abusive language detection.

---

## 🧪 Training and Evaluation

### Training
- Train the models using `train_model.py`:
  ```bash
  python train_model.py --model cnn-lstm

## 📈 Example Usage

- Run the detection script:
   ```bash
   python detect_abusive.py --input sample_comments.txt
   ```
## Output

   ```bash
   Comment: "هذا تعليق تونسي"
   Prediction: Non-abusive
   
   Comment: "كلام بذيء"
   Prediction: Abusive
   ```
## Evaluation

- **Support**: The test dataset contained 348 non-abusive comments (Class 0) and 360 abusive comments (Class 1), for a total of 708 samples.
- **Macro Average**: Averaged precision, recall, and F1-score across both classes.
- **Weighted Average**: Weighted by the number of samples in each class.
  
### CNN-LSTM Performance

The CNN-LSTM model was evaluated on the test set and achieved the following metrics:

| Metric        | Class 0 (Non-abusive) | Class 1 (Abusive) | Average     |
|---------------|-----------------------|-------------------|-------------|
| **Precision** | 0.86                 | 0.89              | 0.88        |
| **Recall**    | 0.89                 | 0.86              | 0.88        |
| **F1-Score**  | 0.88                 | 0.88              | 0.88        |
| **Accuracy**  |                      |                   | **0.88**    |

### BERT Performance

The BERT model was evaluated on the test set and achieved the following metrics:

| Metric        | Class 0 (Non-abusive) | Class 1 (Abusive) | Average     |
|---------------|-----------------------|-------------------|-------------|
| **Precision** | 0.85                 | 0.89              | 0.87        |
| **Recall**    | 0.88                 | 0.85              | 0.87        |
| **F1-Score**  | 0.88                 | 0.86              | 0.87        |
| **Accuracy**  |                      |                   | **0.87**    |


This demonstrates the robustness of the CNN-LSTM  and bert model in detecting abusive comments effectively.

## 📚 Documentation

- [YouTube Data API](https://developers.google.com/youtube/v3)  
- [TensorFlow](https://www.tensorflow.org/)  
- [Transformers by Hugging Face](https://huggingface.co/transformers/)  

---

## 🌟 Contributors

 - Mohamed amine Hassine
 - Baligh Ghaouar
 
---

## 🔒 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

