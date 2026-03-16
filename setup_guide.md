# Setup Guide for Advertisement Success Prediction System

## Prerequisites

1. **Python 3.13** (you have this)
2. **Microsoft Visual C++ Build Tools** (required for NumPy/pandas)

## Step 1: Install Microsoft Visual C++ Build Tools

### Option A: Install via Visual Studio (Recommended)
1. Download Visual Studio Community from: https://visualstudio.microsoft.com/downloads/
2. Run installer
3. Select **"Desktop development with C++"** workload
4. Click Install (approx. 4-6 GB)

### Option B: Install Build Tools only (Smaller download)
1. Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Run installer
3. Select **"Desktop development with C++"** workload
4. Click Install

## Step 2: Install Python Packages

After installing C++ build tools, run:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
python -m pip install -r requirements.txt

# If you still get errors, try installing numpy and pandas separately first
python -m pip install numpy==2.2.4
python -m pip install pandas==2.2.3
python -m pip install -r requirements.txt