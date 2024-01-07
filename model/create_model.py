import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from sklearn import preprocessing
from sklearn import utils

warnings.filterwarnings('ignore')

# Load the pre-trained VGG16 model (or any other model)
base_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features and IDs from a set of images with the same ID
def extract_features_with_id(image_paths, image_id):
    all_features = []
    all_ids = []
    for path in image_paths:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = base_model.predict(img_array).flatten()
        all_features.append(features)
        all_ids.append(image_id)
    return np.array(all_features), np.array(all_ids)

def create_init_model():
    # Sample paths to sets of images with respective IDs
    images_set1 = ['C:/Users/Shan/Desktop/img_processing/images.jpg', 'C:/Users/Shan/Desktop/img_processing/images_1.jpg', 'C:/Users/Shan/Desktop/img_processing/images_2.jpg','C:/Users/Shan/Desktop/img_processing/images_2.jpg','C:/Users/Shan/Desktop/img_processing/images_2.jpg']
    images_set2 = ['C:/Users/Shan/Desktop/img_processing/images4.jpg', 'C:/Users/Shan/Desktop/img_processing/images5.jpg', 'C:/Users/Shan/Desktop/img_processing/images6.jpg', 'C:/Users/Shan/Desktop/img_processing/images6.jpg', 'C:/Users/Shan/Desktop/img_processing/images6.jpg']
    ids = ['SRK', 'SAL']  # IDs corresponding to each set of images

    # Extract features and IDs for each set of images
    features_set1, ids_set1 = extract_features_with_id(images_set1, ids[0])
    features_set2, ids_set2 = extract_features_with_id(images_set2, ids[1])

    # Concatenate the features and IDs from different sets
    all_features = np.concatenate((features_set1, features_set2))
    all_ids = np.concatenate((ids_set1, ids_set2))

    # Train a logistic regression model to predict discrete IDs
    regression_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    regression_model.fit(all_features, all_ids)

    # Save the base model
    joblib.dump(regression_model, 'base_model.pkl')

def update_model(new_images, new_image_id):
    # Extract features and IDs for the new set of images
    features_set3, ids_set3 = extract_features_with_id(new_images, new_image_id)  # Extract features for the new set

    # Load the existing model
    existing_model = joblib.load('base_model.pkl')  # Load your saved model

    # Concatenate the existing model's features and IDs with the new set
    updated_features = np.concatenate((existing_model.coef_, features_set3))
    updated_ids = np.concatenate((existing_model.intercept_ * np.ones(1), ids_set3))  # Adjust the IDs for the new set
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(updated_ids)

    # Retrain the logistic regression model with the updated data
    updated_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    # Use only the features and IDs of the new set, without including existing model's coefficients and intercept
    updated_model.fit(np.concatenate((updated_features, features_set3)), np.concatenate((y_transformed, ids_set3)))

    # Save the updated model
    joblib.dump(updated_model, 'base_model.pkl')

# def predict_id(new_image_path):
#     # loading the updated model to test fetching
#     new_model = joblib.load('base_model.pkl')
#     all_features = []

#     for path in new_image_path:
#         # Extract features from the new image
#         new_image_features = base_model.predict(preprocess_input(np.expand_dims(image.img_to_array(image.load_img(path, target_size=(224, 224))), axis=0)))[0]
#         # all_features.append(new_image_features)
#         if not np.isnan(new_image_features).any():
#             all_features.append(new_image_features)

#     # Combine features for all images into a single representation
#     combined_features = np.mean(all_features, axis=0)  # Use mean, sum, or another aggregation method

#     # Predict the ID for the combined representation using the loaded model
#     predicted_id = new_model.predict(combined_features.reshape(1, -1))[0]

#     return predicted_id

from sklearn.impute import SimpleImputer

def predict_id(new_image_paths):
    # loading the updated model to test fetching
    new_model = joblib.load('base_model.pkl')
    all_features = []

    for path in new_image_paths:
        # Extract features from the new image
        new_image_features = base_model.predict(preprocess_input(np.expand_dims(image.img_to_array(image.load_img(path, target_size=(224, 224))), axis=0)))[0]
        if not np.isnan(new_image_features).any():
            all_features.append(new_image_features)

    if all_features:
        # Combine features for all images into a single representation
        combined_features = np.mean(all_features, axis=0)  # Use mean, sum, or another aggregation method
        
        # Handle NaN values in the combined features using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        combined_features = imputer.fit_transform(combined_features.reshape(1, -1))

        # Ensure there are valid features
        if combined_features.size > 0:
            # Predict the ID for the combined representation using the loaded model
            predicted_id = new_model.predict(combined_features)[0]
            return predicted_id
        else:
            return None  # Handle the case where there are no valid features
    else:
        return None  # Handle the case where all images have NaN features


# create_init_model()