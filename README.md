

<h1>Multi-Model Traffic Flow Prediction <span class="badge">PyTorch</span> <span class="badge">GNN</span></h1>

<p>This project implements and compares three different deep learning architectures for traffic flow prediction using Spatio-Temporal Graph Neural Networks. It supports multiple datasets (PEMS08, HZMetro, SHMetro) and provides a comprehensive training and evaluation pipeline.</p>

<h2>Models Implemented</h2>
<ul>
    <li><strong>STGCN</strong> (Spatio-Temporal Graph Convolutional Network): Combines Chebyshev graph convolution with 2D convolutions and residual connections.</li>
    <li><strong>GAT</strong> (Graph Attention Network): Uses multi-head attention mechanisms to learn adaptive spatial weights combined with temporal convolutions.</li>
    <li><strong>RNN</strong> (Recurrent Neural Network): A baseline LSTM/GRU model that captures temporal dependencies but ignores spatial graph structures.</li>
</ul>

<h2>Supported Datasets</h2>
<p>The notebook features a unified data loader for the following datasets:</p>
<ul>
    <li><strong>PEMS08</strong>: California highway traffic sensors (170 sensors, 5-min intervals).</li>
    <li><strong>HZMetro</strong>: Hangzhou Metro system data.</li>
    <li><strong>SHMetro</strong>: Shanghai Metro system data.</li>
</ul>

<h2>Installation & Requirements</h2>
<p>This project relies on the <strong>PyTorch Geometric</strong> ecosystem. Install the dependencies using the following commands:</p>

<pre><code># 1. Install PyTorch
pip install torch torchvision torchaudio

# 2. Install Graph Neural Network Dependencies
# Note: Ensure the CUDA version matches your PyTorch installation
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# 3. Install Utilities
pip install numpy pandas matplotlib seaborn scikit-learn</code></pre>

<h2>📈 Performance Comparison</h2>
<p>Based on the evaluation results on the SHMetro dataset, the STGCN model demonstrated the best performance:</p>

<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Model</th>
            <th>MAE (vehicles/5min)</th>
            <th>RMSE (vehicles/5min)</th>
            <th>Validation Loss</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1 🏆</td>
            <td><strong>STGCN</strong></td>
            <td>63.65</td>
            <td>150.34</td>
            <td>0.0095</td>
        </tr>
        <tr>
            <td>2</td>
            <td>GAT</td>
            <td>72.39</td>
            <td>169.40</td>
            <td>0.0117</td>
        </tr>
        <tr>
            <td>3</td>
            <td>RNN</td>
            <td>87.11</td>
            <td>175.66</td>
            <td>0.0112</td>
        </tr>
    </tbody>
</table>

<h2>Usage</h2>
<ol>
    <li><strong>Setup Environment</strong>: Run Step 1 to check GPU availability and install libraries.</li>
    <li><strong>Select Dataset</strong>: In Step 2, modify the <code>DATASET_NAME</code> variable to switch between <code>'PEMS08'</code>, <code>'HZMetro'</code>, or <code>'SHMetro'</code>.</li>
    <li><strong>Train Models</strong>: Step 4 runs the multi-model training loop. By default, it trains STGCN, RNN, and GAT sequentially.</li>
    <li><strong>Evaluate</strong>: Step 5 generates performance metrics and comparison plots (Training Loss, MAE Comparison, Radar Charts).</li>
</ol>

<h2>Project Structure</h2>
<ul>
    <li><strong>Step 1</strong>: Environment Setup (Dependencies & GPU)</li>
    <li><strong>Step 2</strong>: Data Loading & Preprocessing (Normalization, Graph Generation)</li>
    <li><strong>Step 3</strong>: Model Architectures (PyTorch `nn.Module` definitions)</li>
    <li><strong>Step 4</strong>: Training Loop (Loss calculation, Backprop, Scheduler)</li>
    <li><strong>Step 5</strong>: Evaluation & Visualization</li>
    <li><strong>Step 6</strong>: Extensions</li>
</ul>
