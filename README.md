# Crop Disease Prediction System

This project is a crop disease prediction system that uses a deep learning model to identify diseases in plant leaves. It also includes a web application built with Streamlit that allows users to upload images of plant leaves and get predictions.

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

The model was trained on the PlantVillage dataset. The dataset should be organized in the following structure:

Plant_Disease_Dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── 00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG
│   │   └── ...
│   └── ...
├── test/
│   └── ...
└── valid/
    └── ...

## Running the Application

To run the Streamlit web application, use the following command:

```bash
streamlit run main.py
```

This will open the application in your web browser.

## Training the Model

The model was trained using the `training_model.ipynb` notebook. To retrain the model, you can run the cells in the notebook. Make sure you have the dataset set up correctly as described above.

The notebook will:
1.  Load the dataset.
2.  Build and compile the CNN model.
3.  Train the model.
4.  Save the trained model as `trained_model.keras`.
5.  Save the training history as `training_hist.json`.

## Dependencies

The main dependencies for this project are:

*   TensorFlow
*   scikit-learn
*   NumPy
*   Matplotlib
*   Seaborn
*   Pandas
*   Streamlit
*   Librosa
*   OpenCV-Python
*   PyMongo

All the required dependencies are listed in the `requirements.txt` file.