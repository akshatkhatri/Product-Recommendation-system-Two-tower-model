# Two-Tower Product Recommendation System

A sophisticated deep learning recommendation system using a Two-Tower neural network architecture that combines collaborative filtering (ALS) with transformer-based embeddings (BERT) to predict user ratings and generate personalized product recommendations on Amazon Products dataset

> 📝 **Read the full technical walkthrough on Medium:** [Building a Two-Tower Recommendation Engine: Combining BERT, ALS, and Popularity-Weighted Negative Sampling](https://medium.com/@akshat.dev/a-walkthrough-of-building-a-personalised-discovery-engine-with-bert-als-and-smart-negative-0e114bae2322?postPublishedType=repub)

**Project Statistics:**
- 🛍️ **600 unique products** from Amazon product reviews
- 👥 **71,044 original reviews** → **284,176 samples** (with negative sampling)
- 🧠 **322,817 trainable parameters** (~1.23 MB model)
- 🚀 **5 iterative model versions** (v1 → v5_power)
- ⚡ **~70ms inference latency** (after warmup)

---

## 📋 Overview

This project implements a **hybrid recommendation engine** that solves the following business problem:

> *Given a user's past review behavior and product metadata, predict which products they are likely to rate highly and recommend personalized top-k products.*

### Key Innovation

The system combines **three powerful techniques**:
1. **Collaborative Filtering** (ALS): Captures user-product interaction patterns through matrix factorization
2. **Content-Based Filtering** (BERT): Extracts semantic meaning from review text and product descriptions
3. **Temporal User Modeling**: Uses running mean of previous reviews to capture evolving user preferences

This hybrid approach handles both **cold-start users** (via content features) and **power users** (via collaborative signals) while avoiding data leakage through careful temporal windowing.

---

## ✨ Features

- **🏗️ Two-Tower Architecture**: Separate embedding towers for users and products with fusion layer for rating prediction
- **🔄 Dual Embedding Strategy**: Combines ALS (64-dim) + BERT (384-dim) embeddings for rich representations
- **🎯 Sophisticated Negative Sampling**: Popularity-weighted sampling (3 negatives per positive) with label smoothing
- **👤 Anonymous User Handling**: Special treatment for users without account information using global statistics
- **📊 Running Mean Embeddings**: Historical user preferences computed from past reviews (excludes current review to prevent leakage)
- **⚡ Efficient Batch Inference**: Pre-computed product embeddings enable fast top-k recommendations
- **💾 Production-Ready**: Models saved in both `.keras` and `.weights.h5` formats
- **🔧 XLA Compilation**: TensorFlow XLA for optimized training and inference

---

## 🏛️ Architecture

### Model Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      TWO-TOWER MODEL                            │
├──────────────────────────┬──────────────────────────────────────┤
│       USER TOWER         │        PRODUCT TOWER                 │
├──────────────────────────┼──────────────────────────────────────┤
│ Inputs:                  │ Inputs:                              │
│  • ALS Embedding (64)    │  • ALS Embedding (64)                │
│  • BERT Embedding (384)  │  • BERT Embedding (384)              │
│  • is_anonymous (1)      │                                      │
│                          │                                      │
│ Total: 449 dims          │ Total: 448 dims                      │
│     ↓                    │     ↓                                │
│ Dense(256, elu)          │ Dense(256, elu)                      │
│     ↓                    │     ↓                                │
│ Dense(128, elu)          │ Dense(128, elu)                      │
│     ↓                    │     ↓                                │
│ Dropout(0.3)             │ Dropout(0.3)                         │
│     ↓                    │     ↓                                │
│ Dense(64, elu)           │ Dense(64, elu)                       │
│     ↓                    │     ↓                                │
└──────────────────────────┴──────────────────────────────────────┘
              ↓                           ↓
              └───────────────┬───────────┘
                              ↓
                    ┌─────────────────────┐
                    │   FUSION LAYER      │
                    │   Concatenate(128)  │
                    │         ↓           │
                    │   Dense(64, elu)    │
                    │         ↓           │
                    │   Dropout(0.2)      │
                    │         ↓           │
                    │   Dense(32, elu)    │
                    │         ↓           │
                    │   Dense(1, linear)  │
                    │         ↓           │
                    │  Rating Prediction  │
                    └─────────────────────┘
```

### Key Design Choices

- **Activation Function**: ELU (Exponential Linear Unit) for all hidden layers
- **Weight Initialization**: He Normal (optimal for ELU activations)
- **Regularization**: Dropout (0.3 in towers, 0.2 in fusion layer)
- **Output Layer**: Linear activation for continuous rating prediction (0-5 scale)
- **Total Parameters**: 322,817 trainable parameters

---

## 📊 Data

### Dataset

The project uses the **Amazon Product Reviews** dataset containing grammar and product reviews.

**Input File**: [GrammarandProductReviews.csv](GrammarandProductReviews.csv)
- 71,044 product reviews
- 600 unique products
- 25 columns including:
  - Product metadata: `id`, `brand`, `categories`, `manufacturer`, `name`
  - Review data: `reviews.rating` (1-5), `reviews.text`, `reviews.title`, `reviews.username`
  - Timestamps: `dateAdded`, `dateUpdated`, `reviews.date`
  - User info: `reviews.userCity`, `reviews.userProvince`

**Generated Files**:
- [negative_samples.csv](negative_samples.csv) - Synthetically generated negative training samples
- [final_df.csv](final_df.csv) - Combined dataset with positive + negative samples (284,176 records)

### Data Statistics

- **Original Reviews**: 71,044
- **After Negative Sampling**: 284,176 (4x expansion)
- **Negative Sample Ratio**: 75% negative samples
- **Mean Rating** (after sampling): 1.85
- **Standard Deviation**: 1.64
- **Rating Range**: 0-5

---

## 🔄 Data Processing Pipeline

### 1. Data Cleaning
- Fill missing review titles with `"No Title"`
- Fill missing review text with `"No Review"`
- Fill missing usernames with placeholder `"Akshat"`
- Fill missing manufacturers with mode value

### 2. Encoding
- Create `user_id` by mapping usernames to integers
- Create `prod_id` by mapping product IDs to integers (0-599)
- Convert ratings to float32

### 3. ALS Collaborative Filtering
- Train AlternatingLeastSquares model using `implicit` library
- **Factors**: 64 dimensions
- Generates:
  - User embeddings (64-dim) from interaction patterns
  - Product embeddings (64-dim) from interaction patterns
- Anonymous users receive zero vectors

### 4. BERT Embeddings

**User Review Embeddings** (384 dimensions):
- Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Input: Concatenate `review_title + review_text`
- **Running Mean Strategy**: Each user receives the average of all their **PREVIOUS** review embeddings
  - Excludes current review (prevents data leakage)
  - First review for new user → zero vector
  - Anonymous users → global average embedding

**Product Embeddings** (384 dimensions):
- Input: Concatenate `name + brand + categories + manufacturer`
- Fixed per product (pre-computed)

### 5. Anonymous User Handling
- Binary feature: `is_anonymous_user` (0 or 1)
- Flags users: "Anonymous", "An anonymous customer", "ByAmazon Customer"
- Allows model to learn different behavior patterns for anonymous users

### 6. Negative Sampling Strategy

**Innovation**: Popularity-based weighted sampling with label smoothing

**Method**:
- For each positive review, generate **k=3** negative samples
- Sample products using popularity distribution (alpha=1.0)
- Assign low negative ratings (uniform random 0-2)
- Ensures model learns from items the user hasn't interacted with
- Prevents popularity bias while maintaining realism

**Result**: Dataset expands from 71,044 → 284,176 records

---

## 💻 Installation

### Requirements

```bash
# Core ML/DL
tensorflow>=2.15.0
keras>=2.15.0
numpy>=1.24.0

# Data Processing
pandas>=2.0.0
scikit-learn>=1.3.0

# Recommendation/Embeddings
implicit>=0.7.0
sentence-transformers>=2.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
pydot>=1.4.0
graphviz>=0.20.0

# Utilities
scipy>=1.11.0
Pillow>=10.0.0
tqdm>=4.65.0
jupyterlab>=4.0.0

# GPU Acceleration (optional)
nvidia-cuda-nvcc-cu12
```

### Setup

```bash
# Clone or navigate to project directory
cd /home/akshat/recsys

# Install dependencies
pip install tensorflow keras numpy pandas scikit-learn implicit sentence-transformers matplotlib seaborn scipy tqdm pydot graphviz

# Launch Jupyter
jupyter lab
```

---

## 🚀 Usage

### 1. Training the Model

Open and run [recsys.ipynb](recsys.ipynb) to:
1. Load and preprocess data
2. Train ALS model
3. Generate BERT embeddings
4. Create negative samples
5. Build and train the Two-Tower model
6. Save trained model

The notebook handles the complete pipeline from raw data to trained model.

### 2. Loading a Trained Model

```python
import tensorflow as tf

# Load the latest model (v5_power)
model = tf.keras.models.load_model('two_tower_recsys_v5_power.keras')

# Or load weights only
# model.load_weights('my_two_tower_model_v5_power.weights.h5')
```

### 3. Getting Recommendations

```python
# Recommend top-5 products for specific users
recommendations = predict_for_users_batched(
    user_ids=[0, 243, 599],
    top_k=5,
    product_ids=None  # None = all products
)

# Output format: {user_id: [(product_id, predicted_rating), ...]}
# Example:
# {
#   0: [(542, 0.948), (120, 0.948), (17, 0.948)],
#   243: [(262, 4.35), (91, 4.34), (28, 4.34)],
#   599: [(91, 4.40), (381, 4.37), (90, 4.36)]
# }
```

### 4. Batch Inference

The system uses **pre-computed product embeddings** for efficient inference:

```python
# Product embeddings are pre-computed and stored
product_tower_forward_pass_embedding_dict  # 600 products ready

# Batch inference with 1024 batch size
# First call: ~66-734ms (warmup)
# Subsequent calls: ~66-75ms (optimized)
```

---

## 📦 Model Versions

The project includes **5 iterative model versions**, each representing improvements and experiments:

| Version | Keras Model | Weights File | Notes |
|---------|-------------|--------------|-------|
| v1 | [two_tower_recsys_v1.keras](two_tower_recsys_v1.keras) | [my_two_tower_model.weights.h5](my_two_tower_model.weights.h5) | Initial baseline |
| v2 | [two_tower_recsys_v2.keras](two_tower_recsys_v2.keras) | [my_two_tower_model_v2.weights.h5](my_two_tower_model_v2.weights.h5) | Iteration 2 |
| v3 | [two_tower_recsys_v3.keras](two_tower_recsys_v3.keras) | [my_two_tower_model_v3.weights.h5](my_two_tower_model_v3.weights.h5) | Iteration 3 |
| v4 | [two_tower_recsys_v4.keras](two_tower_recsys_v4.keras) | [my_two_tower_model_v4.weights.h5](my_two_tower_model_v4.weights.h5) | Iteration 4 |
| v5_power | [two_tower_recsys_v5_power.keras](two_tower_recsys_v5_power.keras) | [my_two_tower_model_v5_power.weights.h5](my_two_tower_model_v5_power.weights.h5) | **Latest & most refined** |

**Recommended**: Use `v5_power` for best performance.

---

## ⚙️ Training Configuration

### Hyperparameters

```python
# Optimizer
optimizer = 'adam'

# Loss & Metrics
loss = 'mean_squared_error'
metrics = ['mean_absolute_error']

# Training
epochs = 10-15
batch_size = 32
total_samples = 284_176
batches_per_epoch = 8_881

# Data Pipeline
shuffle_buffer = 1024
prefetch = tf.data.AUTOTUNE

# Optimization
xla_compilation = True  # Accelerated Linear Algebra
```

### Training Process

1. **Data Pipeline**: TensorFlow `tf.data` with shuffling and prefetching
2. **XLA Compilation**: Compiled clusters for faster execution
3. **Batch Training**: 32 samples per batch, ~8,881 batches per epoch
4. **Monitoring**: MSE loss and MAE metrics tracked during training
5. **Checkpointing**: Models saved after achieving satisfactory performance

**Initial Training Metrics**:
- Starting Loss: ~2.64
- Starting MAE: ~1.24

---

## ⚡ Performance

### Inference Speed

- **First Prediction** (cold start): ~66-734ms
- **Subsequent Predictions** (warm): ~66-75ms
- **Batch Size**: 1024 samples
- **Optimization**: Pre-computed product embeddings stored in memory

### Efficiency Features

1. **Pre-computed Embeddings**: Product tower outputs computed once for all 600 products
2. **Batch Inference**: Tile user features across products for vectorized predictions
3. **XLA Compilation**: JIT compilation for optimized ops
4. **Memory Efficiency**: ~1.23 MB model size

---

## 🎯 Technical Highlights

### Key Innovations

1. **Temporal Data Handling**: Running mean of **PREVIOUS** reviews only (excludes current review)
   - Prevents data leakage during training
   - Captures evolving user preferences naturally

2. **Dual Embedding Strategy**: Combines complementary signals
   - **ALS**: Captures collaborative patterns (who likes similar items)
   - **BERT**: Captures content semantics (what items are similar)

3. **Explicit Anonymous User Feature**: Binary flag allows model to learn different patterns
   - Anonymous users may have different rating behaviors
   - Model can adapt predictions accordingly

4. **Production-Ready Design**:
   - Saved in standard Keras formats (`.keras` and `.weights.h5`)
   - Efficient batch inference pipeline
   - Pre-computed embeddings for scalability

5. **Sophisticated Negative Sampling**:
   - Popularity-weighted (not uniform random)
   - Label smoothing with low ratings (0-2)
   - Realistic training distribution

---

## 📁 Project Structure

```
recsys/
├── README.md                                    # This file
├── recsys.ipynb                                 # Main training notebook
├── dummy.ipynb                                  # Simple data exploration notebook
│
├── GrammarandProductReviews.csv                 # Raw dataset (71K reviews)
├── negative_samples.csv                         # Generated negative samples
├── final_df.csv                                 # Combined dataset (284K samples)
│
├── two_tower_recsys_v1.keras                   # Model v1
├── two_tower_recsys_v2.keras                   # Model v2
├── two_tower_recsys_v3.keras                   # Model v3
├── two_tower_recsys_v4.keras                   # Model v4
├── two_tower_recsys_v5_power.keras             # Model v5 (latest)
│
├── my_two_tower_model.weights.h5               # Weights v1
├── my_two_tower_model_v2.weights.h5            # Weights v2
├── my_two_tower_model_v3.weights.h5            # Weights v3
├── my_two_tower_model_v4.weights.h5            # Weights v4
└── my_two_tower_model_v5_power.weights.h5      # Weights v5 (latest)
```

---

## 🔮 Future Improvements

Potential enhancements for the system:

- [ ] **Multi-task Learning**: Predict both ratings and purchase probability
- [ ] **Attention Mechanisms**: Add attention layers to weight embedding importance
- [ ] **Contextual Features**: Incorporate time-of-day, seasonality, device type
- [ ] **Online Learning**: Support incremental updates with new user interactions
- [ ] **A/B Testing Framework**: Compare model versions in production
- [ ] **Explainability**: Add SHAP or attention visualization for recommendation explanations
- [ ] **Diversity Metrics**: Optimize for recommendation diversity alongside accuracy
- [ ] **Real-time Serving**: Deploy with TensorFlow Serving or TorchServe
- [ ] **Cross-validation**: Implement temporal cross-validation for robust evaluation
- [ ] **Hyperparameter Tuning**: Automated search with Optuna or Ray Tune

---

## 📄 License

This project is available for educational and research purposes.

---

## 👤 Author

**Akshat**  
Data Scientist | Machine Learning Engineer

---

## 🙏 Acknowledgments

- **Dataset**: Amazon Product Reviews (Grammar and Product Reviews subset)
- **Libraries**: TensorFlow, Keras, Hugging Face Transformers, Implicit
- **Inspiration**: Two-Tower architecture from Google Research and YouTube recommendations

---

## 📚 References

- [Two-Tower Models for Recommendation](https://dl.acm.org/doi/10.1145/3298689.3346996)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [Sentence-BERT: Sentence Embeddings using Siamese Networks](https://arxiv.org/abs/1908.10084)
- [Implicit Feedback for Collaborative Filtering](https://dl.acm.org/doi/10.1109/ICDM.2008.22)

---

<div align="center">
  <strong>Built with ❤️ using TensorFlow, Keras, and BERT</strong>
</div>
