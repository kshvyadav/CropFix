# CropFix: Crop Identification & Disease Detection Platform

![CropFix Logo](img/logo.png) (Replace with your project's logo or a screenshot of the main page)

CropFix is an intelligent web platform designed to empower farmers and agricultural practitioners with data-driven insights. It provides a suite of advanced tools for crop identification, disease detection, and crop recommendation, all through an intuitive and user-friendly interface. By integrating state-of-the-art machine learning and interpretable AI, CropFix delivers not only accurate results but also understandable explanations, fostering user trust and facilitating informed action.

## ✨ Core Functionalities

The platform is built around three primary, interconnected modules:

1.  **Advanced Crop Identification and Disease Detection:** Identify crop species and detect potential diseases from an uploaded image.
2.  **Interpretable Disease Detection Explanation:** Understand why a specific disease was predicted using the LIME framework.
3.  **Intelligent Crop Recommendation System:** Get personalized crop suggestions based on soil and environmental parameters.
4.  **Feedback Mechanism:** A crucial user feedback system for continuous improvement.

## 🧠 How It Works

### 1. Advanced Crop Identification and Disease Detection

This module is the central pillar of CropFix, enabling users to identify both crop species and diseases through image analysis.

- **Technology:** The backend is powered by a fine-tuned **ResNet50** deep convolutional neural network. This transfer learning model has been trained on a comprehensive dataset of diverse crop images.
- **Process:** When a user uploads an image, the system preprocesses it and feeds it into the trained ResNet50 model. The model extracts complex features and predicts:
    - **Predicted Crop Name:** The most probable crop name with a confidence score.
    - **Disease Detection:** If visual patterns of a disease are found, the specific disease is identified with a confidence score.
    - **Cure Explanation:** Preliminary guidance on how to prevent or manage the identified disease, including common agricultural practices or treatments.

![Crop Identification and Disease Detection](img/crop_id_input.png)

### 2. Interpretable Disease Detection Explanation using LIME

Understanding the reasoning behind a disease diagnosis is crucial for user trust. This module employs the **Local Interpretable Model-agnostic Explanations (LIME)** framework.

- **Technology:** The LIME model works alongside the ResNet50 prediction to provide insights into the AI's decision-making process.
- **Process:** LIME generates slightly modified versions of the input image and observes how the prediction changes. By analyzing these changes, LIME highlights the specific regions or visual features in the image that were most significant for the prediction.
- **Output:** The system provides a visual explanation, highlighting areas on the leaf or stem (e.g., discoloration patterns) that contributed most significantly to the disease prediction.

![LIME Explanation](img/lime_output.png)

### 3. Intelligent Crop Recommendation System

This module assists users in making informed decisions about crop selection based on their specific conditions.

- **Technology:** The system uses a **Random Forest** model, trained on a comprehensive dataset linking crop suitability to specific agricultural parameters.
- **Process:** When a user inputs parameters like soil pH, Nitrogen (N) content, Phosphorus (P) content, Potassium (K) content, and Temperature, the trained Random Forest model processes this data.
- **Output:** The system generates a ranked list of potential crop suggestions based on the provided conditions.

![Crop Recommendation](img/crop_recommendation_input.png)

### 4. Feedback Mechanism

Farming is ever-evolving, and so is our platform. We've included a simple feedback mechanism to empower users to share their thoughts on the accuracy and utility of our tools. This feedback is crucial for our continuous improvement and for making the platform more reliable and effective for the agricultural community.

## 🚀 Technologies Used

- **Backend:** Python, Flask
- **Machine Learning & Deep Learning:** TensorFlow/Keras (for ResNet50), Scikit-learn (for Random Forest), LIME
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **Image Processing:** OpenCV, Pillow

