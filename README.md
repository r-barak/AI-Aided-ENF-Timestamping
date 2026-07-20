# AI-Aided-ENF-Timestamping

ENF-Timestamping-Project/
│
├── data/                       # Ignored in version control (.gitignore)
│   ├── raw_data/               
│   ├── split_data/             
│   ├── enf_data/               
│   └── models/                 # Saved .pth weights
│
├── src/                        # Core module containing all functions and classes
│   ├── __init__.py
│   ├── config.py               # Hyperparameters and paths
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tcn.py              # Encoder architectures
│   │   └── fusion.py           # Fusion layer architectures
│   ├── data/
│   │   ├── __init__.py
│   │   ├── processing.py       # Audio splitting, resampling, AWGN
│   │   ├── enf_extraction.py   # STFT, bandpass filtering, argmax extraction
│   │   └── dataloader.py       # Triplet mining, Dataset & Loader classes
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py           # InfoNCE, Triplet, CE loss
│   │   └── trainer.py          # Train loops and validation
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # NM, CC, top-K calculations
│       └── evaluate.py         # Embedding generation and evaluation functions
│
├── scripts/                    # Standalone scripts to run the pipeline end-to-end
│   ├── 01_generate_dataset.py  # Runs the data prep pipeline
│   ├── 02_train_encoder.py     # Runs the TCN training
│   ├── 03_train_fusion.py      # Runs the linear fusion training
│   └── 04_evaluate.py          # Runs the final testing and metric tracking
│
├── notebooks/                  
│   └── tutorial.ipynb          # The cleaned-up guideline notebook
│
├── requirements.txt            # Python dependencies (torch, scipy, librosa, etc.)
└── README.md                   # Setup instructions and project overview


## Getting Started

### 1. Environment Setup
First, clone the repository and install the required Python dependencies:
\`\`\`bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt (add "requests" to it)
\`\`\`

### 2. Data Preparation
This project relies on the H1 subset of the ENF-WHU-Dataset. We have provided an automated script to download and structure the data for you.

Run the following command from the root directory of the repository to fetch the data:
\`\`\`bash
python scripts/download_data.py
\`\`\`
*Note: This will create a `H1_Dataset/` folder in your directory containing all the necessary audio files.*

### 3. Running the Model
Once the data is downloaded, you can train the model using:
\`\`\`bash
python train.py --data_dir ./H1_Dataset
\`\`\`