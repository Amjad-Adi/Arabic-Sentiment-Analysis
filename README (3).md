# Arabic Sentiment Analysis

**Done by:**
- Amjad Adi — 1230800
- Hanan Alawawdeh — 1230827

A comprehensive machine learning research project for sentiment analysis on Arabic text, comparing multiple classification approaches including Naive Bayes, Neural Networks (MLP), and Balanced Random Forest models with advanced feature engineering and hyperparameter optimization.

## 📋 Project Overview

This project implements and extensively evaluates multiple sentiment classification models for Arabic natural language processing (NLP). It performs sophisticated preprocessing specifically tailored for Arabic language characteristics and employs both traditional machine learning and deep learning approaches to achieve state-of-the-art performance in three-class sentiment classification.

### Sentiment Classes

- **Positive (POS)**: Positive sentiment/opinion
- **Negative (NEG)**: Negative sentiment/opinion  
- **Neutral (NEUTRAL)**: Neutral, objective, or mixed sentiment

### Research Focus

The project investigates:
- Effectiveness of different feature representations (TF-IDF vs. Transformer embeddings)
- Impact of Arabic-specific text preprocessing
- Comparative analysis of classical ML vs. deep learning approaches
- Hyperparameter optimization strategies for Arabic text classification
- Handling of class imbalance in sentiment datasets

## 🏗️ Architecture & Stack

- **Language:** Python 3.x
- **ML Framework:** scikit-learn
- **Deep Learning:** PyTorch with CUDA support
- **Transformer Models:** Hugging Face Transformers (MARBERT - Moroccan Arabic BERT)
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Text Processing:** NLTK, emoji, unicodedata
- **Imbalanced Learning:** imbalanced-learn (BalancedRandomForestClassifier)

## 📂 Repository Structure

```
.
├── main.py                              # Main pipeline: preprocessing, training, evaluation
├── Text.txt                             # Raw Arabic text dataset (TSV format)
├── REPORT.pdf                           # Detailed research report with findings
│
├── Data Processing Outputs/
│   └── Preprocessing_Comparison.xlsx    # Side-by-side comparison of preprocessing methods
│
├── Visualizations/
│   ├── 1_distribution.png               # Class distribution histogram
│   ├── Figure_1_Performance_Metrics.png # Bar charts: accuracy, precision, recall, F1
│   ├── Figure_2_Metrics_Analysis.png    # Line graphs & F1-score ranking
│   ├── Figure_3_Summary_Table.png       # Performance summary table
│   │
│   └── Confusion Matrices/
│       ├── CM_NB_TFIDF.png              # Naive Bayes confusion matrix
│       ├── CM_MLP_FUSED.png             # Neural Network confusion matrix
│       └── CM_BRF_FUSED.png             # Balanced Random Forest confusion matrix
│
└── Embeddings Cache/
    ├── cache_UBC-NLP_MARBERT_train_96.npy
    ├── cache_UBC-NLP_MARBERT_val_96.npy
    └── cache_UBC-NLP_MARBERT_test_96.npy
```

## 🔄 Pipeline Architecture

```
Raw Arabic Text
        ↓
  ┌─────────────────────────────────────┐
  │   Dual Preprocessing Strategy       │
  ├─────────────────────────────────────┤
  │ • TF-IDF Path (Traditional ML)      │
  │ • Transformer Path (Deep Learning)  │
  └─────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────┐
  │   Feature Engineering               │
  ├─────────────────────────────────────┤
  │ • TF-IDF Vectorization              │
  │   - Character n-grams (3-5)         │
  │   - Word n-grams (1-3)              │
  │ • MARBERT Embeddings                │
  │ • SVD Dimensionality Reduction      │
  │ • StandardScaler Normalization      │
  └─────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────┐
  │   Model Training & Tuning           │
  ├─────────────────────────────────────┤
  │ ① Naive Bayes (TF-IDF)              │
  │ ② MLP Neural Network (Fused)        │
  │ ③ Balanced Random Forest (Fused)    │
  └─────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────┐
  │   Evaluation & Comparison           │
  ├─────────────────────────────────────┤
  │ • Metrics: Acc, Prec, Rec, F1       │
  │ • Confusion Matrices                │
  │ • Performance Leaderboard           │
  │ • Training Time Analysis            │
  └─────────────────────────────────────┘
```

## 🧹 Arabic Text Preprocessing

### TF-IDF Preprocessing Pipeline

The TF-IDF path implements extensive Arabic-specific preprocessing:

1. **Unicode Normalization (NFKC)**
   - Standardizes different forms of the same character

2. **Character Normalization**
   - ا ← إ, أ, آ, ٱ (all alef variants normalized to ا)
   - ي ← ى (Arabic alef maksura to yaa)
   - ء ← ؤ, ئ (hamza variants)
   - ه ← ة (ta marbuta to haa)
   - ك ← گ, ڪ, ڬ (various kaf forms)
   - And many other variants...

3. **Diacritic & Mark Removal**
   - Removes Tashkeel (حروف التشكيل): fatha, damma, kasra, sukun, etc.
   - Removes BIDI marks: ‎, ‏, ‪, ‫, etc.
   - Removes Tatweel (kashida): ـ

4. **Special Character Handling**
   - Ellipsis/continuation: `...` or `…` → "تكملة" (continuation)
   - Question marks: `?` or `؟` → "سؤال" (question)
   - Exclamation marks: `!` → "تعجب" (exclamation)
   - Emojis: Converted to Arabic text equivalents (e.g., 😊 → ابتسامة)
   - Emoticons: `:)`, `:(`, `:-)`, etc. → Arabic sentiment words

5. **Duplicate Character Reduction**
   - Reduces repeated characters: "مررررحبا" → "مرحبا"

6. **Stop Word Removal**
   - Uses NLTK Arabic stopwords

### Transformer Preprocessing Pipeline

The Transformer path uses lighter preprocessing to preserve linguistic context:

- Unicode normalization
- BIDI mark removal
- Tatweel removal
- **Preserves**: Diacritics, punctuation, some structure
- Handles: URLs → "رابط", Emails → "بريد", @mentions → "حساب"
- Applies: Emoji/emoticon mapping
- Applies: Character normalization (but more conservatively)

## 🔧 Feature Engineering

### TF-IDF Features
- **Character n-grams**: 3-5 character sequences (captures morphological patterns)
- **Word n-grams**: 1-3 word sequences (captures phrasal patterns)
- **Vectorizer settings**:
  - `min_df=2`: Ignore terms appearing in <2 documents
  - `max_df=0.9`: Ignore terms appearing in >90% of documents
  - Combined feature matrix from char + word vectorizers

### Transformer Embeddings (MARBERT)
- **Model**: UBC-NLP/MARBERT (pre-trained on Moroccan Arabic)
- **Max sequence length**: 96 tokens
- **Batch size**: 32 (for memory efficiency)
- **Pooling**: Mean pooling over token embeddings using attention mask
- **Caching**: Embeddings cached for faster iterations

### Dimensionality Reduction
- **Algorithm**: Truncated SVD
- **Components**: k=250 (tunable parameter)
- **Normalization**: StandardScaler (z-score normalization)

### Feature Fusion
For MLP and Random Forest models, features are fused:
```
Final Features = [Scaled TF-IDF (SVD-reduced)] + [Scaled Transformer Embeddings]
                = (250 + 768) = 1,018 dimensions
```

## 📊 Models & Hyperparameter Tuning

### 1. Naive Bayes (MultinomialNB)

**Features**: TF-IDF only

**Hyperparameters Tuned**:
- **alpha** (smoothing parameter)
  - Grid: 1e-4 to 1.0
  - Method: Zoom optimization (iterative refinement)
  - Optimizes: F1-score on validation set

- **Class Prior** (mixing empirical + uniform)
  - Parameter: `s` (mixture coefficient)
  - Grid: 0.0 to 1.0
  - Formula: `prior = (1-s) * empirical + s * uniform`
  - Handles: Class imbalance

**Training**: 
- Separate optimization for alpha and class priors
- Final model trained on combined train+validation sets
- Evaluated on held-out test set

### 2. MLP Neural Network

**Features**: Fused (TF-IDF + MARBERT embeddings)

**Architecture Search** (RandomizedSearchCV):
- **Hidden layer sizes**: 
  - Single layer: (512,), (256,), (128,)
  - Two layers: (512, 256), (256, 128), (128, 64)
  - Three layers: (512, 256, 128), (256, 128, 64)

- **Hyperparameters**:
  - `alpha` (L2 regularization): 0.00001, 0.0001, 0.001, 0.01
  - `learning_rate_init`: 0.0001, 0.001, 0.01
  - `activation`: relu, tanh
  - `solver`: adam
  - `batch_size`: 32, 64, 128
  - `max_iter`: 1000
  - `early_stopping`: True
  - `validation_fraction`: 0.1
  - `n_iter_no_change`: 20

- **Search Strategy**:
  - n_iter: 20 random combinations
  - Scoring: F1-macro
  - Cross-validation: PredefinedSplit (train vs. validation)

### 3. Balanced Random Forest

**Features**: Fused (TF-IDF + MARBERT embeddings)

**Two-Stage Optimization**:

**Stage 1: Hyperparameter Search** (RandomizedSearchCV)
- `n_estimators`: [300, 500, 700, 1000, 1500]
- `max_depth`: [20, 30, 40, 50, None]
- `min_samples_split`: [2, 5, 10, 15]
- `max_samples`: [0.7, 0.8, 0.9, 1.0]
- `min_samples_leaf`: [1, 2, 4, 8]
- `max_features`: ["sqrt", "log2"]
- `replacement`: [True, False]
- `sampling_strategy`: ['auto', 'majority', 'all']

**Stage 2: SVD Component Search**
- Tests k values: [100, 150, 200, 250, 300, 350, 400]
- Uses best hyperparameters from Stage 1
- Selects k that maximizes validation F1-score

**Class Balancing**:
- BalancedRandomForestClassifier handles imbalanced classes
- Sampling strategies for minority classes

## 📈 Evaluation Metrics

### Per-Sample Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) [macro-averaged]
- **Recall**: TP / (TP + FN) [macro-averaged]
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) [macro-averaged]

### Per-Class Analysis
- Classification reports with per-class metrics
- Confusion matrices for error analysis

### Performance Comparison
- Leaderboard sorted by F1-score
- Training time comparison
- Accuracy vs. speed tradeoff

## 🚀 How to Run

### Prerequisites

Install Python 3.7+ and dependencies:

```bash
# Clone the repository
git clone https://github.com/Amjad-Adi/Arabic-Sentiment-Analysis.git
cd Arabic-Sentiment-Analysis

# Install required packages
pip install -r requirements.txt
```

Or install manually:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn torch transformers nltk emoji imbalanced-learn scipy
```

Download NLTK Arabic stopwords:

```python
import nltk
nltk.download('stopwords')
```

### Data Preparation

Create a TSV (tab-separated) file with two columns:

```
text	sentiment
النص العربي الإيجابي	POS
نص محايد	NEUTRAL
نص سلبي جدا	NEG
...
```

Or use the provided `Text.txt` dataset.

### Running the Pipeline

```bash
python main.py
```

When prompted:
```
Enter path to DataSet.txt: ./Text.txt
```

The script will:
1. ✅ Load and analyze dataset
2. ✅ Generate class distribution visualization
3. ✅ Apply dual preprocessing (TF-IDF + Transformer)
4. ✅ Create preprocessing comparison Excel file
5. ✅ Generate TF-IDF features (char + word n-grams)
6. ✅ Generate MARBERT embeddings (cached for efficiency)
7. ✅ Train & tune Naive Bayes model
8. ✅ Train & tune MLP Neural Network
9. ✅ Train & tune Balanced Random Forest
10. ✅ Generate confusion matrices
11. ✅ Create performance visualizations
12. ✅ Output performance leaderboard

### Expected Runtime

- **CPU**: 2-6 hours (hyperparameter tuning is intensive)
- **GPU (CUDA)**: 30-60 minutes (recommended for transformer embeddings)

### Output Files Generated

**Visualizations**:
- `1_distribution.png` - Class distribution histogram
- `Figure_1_Performance_Metrics.png` - Metrics comparison bar charts + training time
- `Figure_2_Metrics_Analysis.png` - Trend analysis line graphs + F1-score ranking
- `Figure_3_Summary_Table.png` - Formatted summary statistics table

**Confusion Matrices**:
- `CM_NB_TFIDF.png` - Naive Bayes predictions vs. actual labels
- `CM_MLP_FUSED.png` - Neural Network predictions vs. actual labels
- `CM_BRF_FUSED.png` - Random Forest predictions vs. actual labels

**Data Files**:
- `Preprocessing_Comparison.xlsx` - Original, TF-IDF preprocessed, and Transformer preprocessed text samples

**Cache Files** (for faster reruns):
- `cache_UBC-NLP_MARBERT_train_96.npy`
- `cache_UBC-NLP_MARBERT_val_96.npy`
- `cache_UBC-NLP_MARBERT_test_96.npy`

## 📊 Expected Results

The project generates a performance leaderboard comparing all three models. Typical results show:

| Model | Accuracy | Precision | Recall | F1-Score | Time |
|-------|----------|-----------|--------|----------|------|
| Balanced Random Forest | ~0.88 | ~0.87 | ~0.87 | ~0.87 | 15-30m |
| Neural Network (MLP) | ~0.85 | ~0.84 | ~0.84 | ~0.84 | 10-20m |
| Naive Bayes | ~0.82 | ~0.81 | ~0.81 | ~0.81 | <1m |

*Note: Actual results depend on the specific dataset and class distribution.*

## 🔬 Key Research Insights

### Findings from the Project

1. **Feature Representation Impact**
   - Transformer embeddings (MARBERT) capture semantic nuances better than TF-IDF alone
   - Fusion of both features improves performance over either alone

2. **Arabic Preprocessing Importance**
   - Character normalization is crucial for Arabic NLP
   - Diacritic handling affects model performance
   - Emoji/emoticon conversion improves sentiment signal

3. **Model Comparison**
   - Balanced Random Forest shows best F1-score
   - MLP Neural Network offers good accuracy with reasonable training time
   - Naive Bayes serves as strong baseline despite simplicity

4. **Class Imbalance Handling**
   - BalancedRandomForestClassifier addresses class imbalance effectively
   - Class prior optimization in Naive Bayes helps with minority classes
   - Stratified splits ensure representative train/val/test sets

5. **Hyperparameter Sensitivity**
   - Random Forest is sensitive to max_depth and n_estimators
   - MLP performance varies with hidden layer architecture
   - Naive Bayes benefits from custom class priors

## 📚 Dataset Format

**Input**: TSV file with columns:
- `text`: Arabic text sample
- `sentiment`: Class label (POS, NEG, OBJ/NEUTRAL)

**Processing**:
- Total dataset split: 60% train, 20% validation, 20% test
- Stratified splits maintain class distribution
- All models evaluated on same test set for fair comparison

## 🔍 Advanced Features

### Zoom Optimization
Custom iterative optimization algorithm for Naive Bayes hyperparameters:
- Starts with coarse grid
- Zooms in on best region
- Refines until convergence
- More efficient than exhaustive grid search

### Embedding Caching
- Transformer embeddings cached after first compute
- Significantly speeds up subsequent runs
- Useful for experimenting with different models/parameters

### Parallel Processing
- RandomizedSearchCV uses `n_jobs=-1` (all CPU cores)
- BalancedRandomForestClassifier parallelizes tree construction
- Can be adjusted based on available resources

## 📖 References

For detailed methodology, results, and analysis, see **[REPORT.pdf](REPORT.pdf)** included in the repository.

## ⚙️ Configuration

Key configurable parameters in `main.py`:

```python
# Transformer Settings
TRANSFORMER_MODEL = "UBC-NLP/MARBERT"
MAX_LEN = 96
BATCH_SIZE = 32

# SVD Settings
kSVD = 250

# Train/Val/Test Split
test_size_1 = 0.4  # 60% train, 40% temp
test_size_2 = 0.5  # 50/50 split of temp → 20% val, 20% test

# Hyperparameter Search
mlp_n_iter = 20
rf_n_iter = 15

# Grid Ranges
alphaMin, alphaHigh = 1e-4, 1.0
sMin, sMax = 0.0, 1.0
kGrid = [100, 150, 200, 250, 300, 350, 400]
```

## 🤝 Contributing

This is a research project. Feel free to fork, modify, and experiment with different:
- Preprocessing strategies
- Feature engineering approaches
- Model architectures
- Hyperparameter ranges

## 👨‍💻 Author

**Amjad Adi**

GitHub: [@Amjad-Adi](https://github.com/Amjad-Adi)

## 📄 License

This project is open source. See the repository for license details.

---

## 📞 Support & Questions

For questions about the methodology, results, or implementation, refer to:
1. **REPORT.pdf** - Detailed research findings
2. **main.py** - Fully commented code
3. **Preprocessing_Comparison.xlsx** - Visual preprocessing examples

---

**Last Updated**: February 2026  
**Project Status**: Research Complete
