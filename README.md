# 🌍 CNN-Based Waste Classification Model

This project is a **Convolutional Neural Network (CNN)-based waste classification system** that categorizes waste into **Recyclable and Organic Waste** using deep learning. It was developed as part of my **AICTE Internship** in collaboration with **Edunet Foundation and Shell**.

## 🚀 Project Overview
Waste misclassification is a major challenge in waste management, leading to inefficient recycling and environmental hazards. This project leverages **Deep Learning (CNNs)** to automate waste classification based on images, improving waste segregation efficiency.

## 🎯 Learning Objectives
- Understand the role of **CNNs in image classification**.
- Develop a **waste classification model** using **TensorFlow & Keras**.
- Optimize **model training on Google Colab** for better accuracy.
- Analyze **model predictions and challenges** in waste classification.

## 🛠️ Tools & Technologies Used
- **Programming Language**: Python 🐍  
- **Deep Learning Libraries**: TensorFlow & Keras 🤖  
- **Image Processing**: OpenCV 📷  
- **Data Handling**: NumPy & Pandas 📊  
- **Model Training**: **Google Colab** ☁️ *(with GPU acceleration)*
- **Dataset**: Techsash Waste Classification Dataset (from Kaggle)  

## 📌 Methodology
1. **Dataset Collection & Preprocessing**
   - Images resized and normalized for CNN input.
   - Data augmentation applied to enhance model generalization.

2. **CNN Model Architecture**
   - **Convolutional Layers** for feature extraction.
   - **ReLU Activation** for non-linearity.
   - **Pooling Layers** to reduce dimensionality.
   - **Fully Connected Layer (Dense)** for classification.

3. **Training & Optimization**
   - Model trained on **Google Colab** with different epoch values.
   - **Adam Optimizer** and **Categorical Cross-Entropy Loss** used.
   - Hyperparameter tuning performed for better accuracy.

4. **Evaluation & Testing**
   - Accuracy and loss graphs analyzed.
   - Model tested on new images to assess classification performance.

## 🏆 Results & Observations
✔️ The model successfully classifies waste images into **Recyclable** and **Organic** categories.  
✔️ Reducing the number of epochs helped prevent overfitting and improved generalization.  
✔️ Some misclassification occurs due to **dataset limitations and similarities in waste types**.  
✔️ Future improvements can include **fine-tuning the model, expanding the dataset, or using transfer learning**.  

## 📥 Installation & Usage
### **Run the Project on Google Colab**
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/elamaran25/CNN-model-to-classify-images-of-plastic-waste-into-different-categories.git
   cd CNN-model-to-classify-images-of-plastic-waste-into-different-categories
   ```

2. **Open Google Colab & Upload Notebook**  
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `Waste_classification.ipynb`
   - Connect to a **GPU runtime** *(recommended for faster training).*

3. **Install Dependencies in Colab**  
   Run the following in a Colab cell:  
   ```python
   !pip install tensorflow opencv-python numpy pandas
   ```

4. **Train the Model & Predict**
   - Run all notebook cells sequentially.
   - Upload an image to test the classification.

## 🔥 Improvements & Future Scope
Since this project follows a structured model provided by my internship mentor, I focused on:
- Understanding **CNN-based image classification**.
- Adjusting **training epochs** to optimize model accuracy.
- Identifying areas for **future improvements**, including better dataset balance and fine-tuning.

### **Potential Future Enhancements:**
🚀 **Expand the dataset** for improved accuracy.  
🚀 **Implement Transfer Learning** using pre-trained models (e.g., VGG16, ResNet).  
🚀 **Deploy the model** on **IoT-enabled Smart Bins** for real-world waste management.  

## 🏅 Internship Details
This project was developed as part of my **AICTE Internship** with:  
- **Edunet Foundation**  
- **Shell AICTE Future Skills Initiative**  

## 📜 License
This project is licensed under the **MIT License**, allowing anyone to use and modify it for educational and research purposes.

---

### 📩 Connect with Me
For any questions or suggestions, feel free to connect:  
🔗 **LinkedIn**: [elamaran25](https://www.linkedin.com/in/elamaran25/) 



  

