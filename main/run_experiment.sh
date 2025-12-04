#!/bin/bash

# Federated Learning Experiment Runner
# This script runs the complete experiment and generates plots

echo "=================================="
echo "Federated Learning Experiment"
echo "Pneumonia Detection - 5 Hospitals"
echo "=================================="
echo ""

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the trial-1 directory"
    exit 1
fi

# Step 1: Install dependencies
echo "📦 Step 1: Installing dependencies..."
pip install -e . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Step 2: Clean previous results
if [ -f "metrics_history.json" ]; then
    echo "🧹 Cleaning previous results..."
    rm -f metrics_history.json training_results.png
    echo "✅ Previous results cleaned"
    echo ""
fi

# Step 3: Run federated learning
echo "🚀 Step 2: Starting federated learning (20 rounds)..."
echo "⏱️  This will take approximately 10-30 minutes depending on your hardware"
echo ""
flwr run .

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
else
    echo ""
    echo "❌ Training failed or was interrupted"
    exit 1
fi

# Step 4: Generate plots
if [ -f "metrics_history.json" ]; then
    echo "📊 Step 3: Generating plots..."
    python plot_results.py
    
    if [ -f "training_results.png" ]; then
        echo ""
        echo "=================================="
        echo "✅ EXPERIMENT COMPLETE!"
        echo "=================================="
        echo ""
        echo "Generated files:"
        echo "  📄 metrics_history.json - Raw metrics data"
        echo "  📈 training_results.png - Visualization"
        echo ""
        echo "To view the plot:"
        echo "  open training_results.png    (macOS)"
        echo "  xdg-open training_results.png (Linux)"
        echo ""
    else
        echo "⚠️  Plot generation failed"
    fi
else
    echo "⚠️  No metrics file found. Training may not have completed properly."
fi
