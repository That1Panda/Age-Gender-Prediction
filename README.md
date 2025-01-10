# Age and Gender Prediction using Transfer Learning  

## Overview  

This project predicts **age** and **gender** from facial images using **transfer learning** with the VGG16 model. The task involves **multi-objective learning**, where both predictions are made simultaneously. Each task is associated with a specific cost function:  

- **Age Prediction**: Regression task using Mean Squared Error (MSE).  
- **Gender Prediction**: Classification task using Binary Cross-Entropy (BCE).  

The objectives are weighted equally during training, with \(\alpha = 1\) and \(\beta = 1\).  

The entire development process, including model design, training, and evaluation, is documented in a **Jupyter Notebook**. The notebook includes:  
1. Model architecture and training steps.  
2. Visualized results for better understanding.  
3. Examples for testing the model's accuracy.  

This project was guided by the book **Modern Computer Vision with PyTorch, Second Edition**.  

---

## Features  

- **Transfer Learning**: Leverages the pre-trained VGG16 model for feature extraction.  
- **Multi-Objective Framework**: Utilizes separate heads for age regression and gender classification.  
- **FairFace Dataset**: A balanced dataset ensuring diversity and fairness in model predictions.  
- **In-Notebook Visualizations**: Displays training metrics and example predictions.  

---

## Dataset  

The model is trained on the **FairFace** dataset, which contains labeled images with:  
- **Age**: Continuous numerical values.  
- **Gender**: Binary classification (Male/Female).  

FairFace is chosen for its balanced representation across age groups, genders, and ethnicities.  

---

## How It Works  

1. **Model Architecture**  
   - **Backbone**: VGG16 (pre-trained on ImageNet).  
   - **Task-Specific Heads**:  
     - **Age Regression**: Fully connected layers with MSE loss.  
     - **Gender Classification**: Fully connected layers with BCE loss.  

2. **Training**  
   - Total Loss = \( \text{MSE Loss} + \text{BCE Loss} \).  
   - Optimizer: Adam with learning rate scheduling.  
   - Training conducted in Google Colab for ease of use and GPU acceleration.  

3. **Evaluation**  
   - **Age Prediction Metric**: Mean Absolute Error (MAE).  
   - **Gender Prediction Metric**: Accuracy.  

4. **Testing**  
   - Model predictions are tested on sample images.  
   - Outputs include:  
     - Predicted age and gender.  
     - True labels for comparison.  

---

## Results  

- **Age Prediction**: Achieved a Mean Absolute Error (MAE) of **6.3 years**.  
- **Gender Prediction**: Reached an accuracy of **84%**.  

The notebook includes visualizations comparing predicted and true values, along with example results for clarity.  

---

## How to Use  

1. **Open the Notebook in Google Colab**  
   - Upload the `Age_Gender_Prediction.ipynb` notebook to Google Colab.  

2. **Follow the Instructions in the Notebook**  
   - Train the model using the provided dataset.  
   - Evaluate the model's performance.  
   - Test the model on custom images.  

---

## Future Work  

- Fine-tune VGG16 for improved accuracy.  
- Experiment with alternate weighting strategies for multi-objective loss.  
- Extend the model to predict additional attributes (e.g., race, emotion).  

---

## License  

This project is licensed under the MIT License.  

## Acknowledgments  

- Pre-trained VGG16 model used from [TensorFlow/Keras](https://keras.io/) or [PyTorch](https://pytorch.org/).  
- Dataset: [FairFace Dataset](https://github.com/joojs/fairface).  
- Guided by: **Modern Computer Vision with PyTorch, Second Edition**.  