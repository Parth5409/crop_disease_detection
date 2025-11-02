
# Project Documentation: Crop Disease Prediction System

## 1. Project Overview

The Crop Disease Prediction System is a full-stack application designed to assist farmers and agricultural enthusiasts in identifying plant diseases from images of plant leaves. The system uses a deep learning model to predict the disease and provides information on preventions and remedies. It also features a community forum for users to discuss agricultural topics.

### Key Features:

*   **AI-Powered Disease Prediction:** Upload an image of a plant leaf to get an instant disease prediction.
*   **Disease Information:** Access detailed information on preventions and remedies for a wide range of plant diseases.
*   **Agri Community:** A forum for users to ask questions, share knowledge, and interact with other members of the agricultural community.
*   **User Authentication:** Secure user registration and login system.
*   **Profile Management:** Users can manage their profile information.

## 2. Machine Learning Model

The core of the project is a Convolutional Neural Network (CNN) built with TensorFlow and Keras. This model is trained to classify images of plant leaves into 38 different disease categories, including healthy leaves.

### Model Architecture (`train.py`)

The CNN architecture is defined in `train.py` and consists of the following layers:

*   **Convolutional Layers:** The model uses a series of `Conv2D` layers with `relu` activation. These layers are responsible for extracting features from the input images. The number of filters increases in deeper layers (32, 64, 128, 256, 512) to capture more complex patterns.
*   **Max Pooling Layers:** `MaxPool2D` layers are used after the convolutional layers to reduce the spatial dimensions of the feature maps, which helps to make the model more efficient and to control overfitting.
*   **Dropout Layers:** `Dropout` layers are included to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
*   **Flatten Layer:** A `Flatten` layer is used to convert the 2D feature maps into a 1D vector.
*   **Dense Layers:**
    *   A fully connected `Dense` layer with 1500 units and `relu` activation.
    *   The final output layer is a `Dense` layer with 38 units (corresponding to the 38 classes of diseases) and a `softmax` activation function to output a probability distribution over the classes.

### Training (`train.py`)

*   **Dataset:** The model is trained on the `Plant_Disease_Dataset`, which is split into `train` and `valid` sets.
*   **Data Preprocessing:** The `tf.keras.utils.image_dataset_from_directory` function is used to load and preprocess the images. The images are resized to 128x128 pixels and converted to RGB.
*   **Compilation:** The model is compiled using the `Adam` optimizer with a learning rate of 0.0001, `categorical_crossentropy` as the loss function, and `accuracy` as the evaluation metric.
*   **Training:** The model is trained for 10 epochs using the `model.fit()` method. The training history, including accuracy and loss, is saved to `training_hist.json`.
*   **Saved Model:** The trained model is saved as `trained_model.keras`.

### Prediction (`main.py`)

The `model_prediction` function in `main.py` handles the prediction process:

1.  **Load Model:** The `trained_model.keras` file is loaded.
2.  **Image Preprocessing:** The uploaded image is resized to 128x128 pixels and converted to a NumPy array.
3.  **Prediction:** The `model.predict()` method is called to get the prediction probabilities for each class.
4.  **Get Result:** `np.argmax()` is used to determine the class with the highest probability, which is the predicted disease.

## 3. Streamlit Web Application (`main.py`)

The user interface is a web application built with Streamlit.

### Pages and Functionality:

*   **Home Page:**
    *   Allows users to upload an image of a plant leaf.
    *   A "Predict" button triggers the disease prediction.
    *   Displays the predicted disease, confidence score, and the top 3 predictions.
    *   Provides "View Preventions" and "View Remedies" buttons that fetch and display information from the MongoDB database.

*   **Agri Community Page:**
    *   A forum where logged-in users can post questions and answer questions from other users.
    *   Discussions are stored and retrieved from the `agri_community` collection in MongoDB.

*   **Settings Page:**
    *   **Account Management:** Provides login and sign-up functionality. User credentials are stored in the `users` collection in MongoDB.
    *   **Profile Management:** Logged-in users can update their profile information, which is stored in the `profiles` collection.
    *   **Farm Store Settings & Community Settings:** These are currently placeholder sections.

### Styling:

The application uses custom CSS injected via `st.markdown(unsafe_allow_html=True)` to create a visually appealing and user-friendly interface.

## 4. Database Integration

The application uses MongoDB to store and manage data.

*   **Connection:** The application connects to a local MongoDB instance at `mongodb://localhost:27017/`.
*   **Database:** `NewDataBase`

### Collections:

*   **`SampleCollection`:**
    *   Populated by `initialize_db.py`.
    *   Stores detailed information about each disease, including `prevention` and `remedies`.
    *   This collection is queried to provide users with information about the predicted disease.

*   **`agri_community`:**
    *   Stores questions and answers from the Agri Community forum.
    *   Each document contains the user's email, the question, a list of answers, and a timestamp.

*   **`users`:**
    *   Stores user authentication information (name, email, and password).

*   **`profiles`:**
    *   Stores additional user profile information, such as name, email, and address.

### Scripts:

*   **`initialize_db.py`:** This script should be run once to populate the `SampleCollection` with the disease management data.
*   **`dp_connection.py`:** A simple script for testing the connection to the MongoDB database.

## 5. How It All Works Together

1.  **Initialization:** The `initialize_db.py` script is run to set up the database with disease information.
2.  **User Interaction:** A user opens the Streamlit application in their web browser.
3.  **Prediction:**
    *   The user uploads a leaf image on the Home page and clicks "Predict".
    *   The `model_prediction` function in `main.py` processes the image and uses the trained model to predict the disease.
    *   The result is displayed to the user.
4.  **Information Retrieval:**
    *   The user can click on "View Preventions" or "View Remedies".
    *   The application queries the `SampleCollection` in MongoDB to retrieve and display the requested information.
5.  **Community Interaction:**
    *   Users can log in or sign up on the Settings page.
    *   Logged-in users can participate in the Agri Community by asking and answering questions.
