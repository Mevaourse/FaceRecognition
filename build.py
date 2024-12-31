import os
import numpy as np
import cv2
import face_recognition
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Paths to your dataset and output directory
train_path = 'Dataset/train/'
val_path = 'Dataset/val/'
output_faces_dir = 'output_faces/'  # Directory to save detected faces

# Create the output directory if it doesn't exist
os.makedirs(output_faces_dir, exist_ok=True)

# Function to get labels and images from a given directory
def load_dataset(path):
    labels = os.listdir(path)
    X = []
    y = []
    for i, label in enumerate(labels): 
        img_filenames = os.listdir(os.path.join(path, label))
        for filename in img_filenames:
            filepath = os.path.join(path, label, filename)
            img = cv2.imread(filepath)
            
            # Ignore if no face is found in the image
            try:
                encode = preprocess(img, label, filename)
            except Exception as e:
                print(e, ":", label, filename)
                continue
            
            X.append(encode)
            y.append(i)
    return np.asarray(X), np.asarray(y)

# Preprocess function
def preprocess(img, label, filename):
    # Detect face in the image
    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        raise Exception("No face found")
    
    # Use the first detected face
    (t, r, b, l) = face_locations[0]
    face_img = img[t:b, l:r]
    
    # Resize the face image to a standard size
    face_img = cv2.resize(face_img, (224, 224))
    
    # Save the detected face image
    save_path = os.path.join(output_faces_dir, f"{label}_{filename}")
    cv2.imwrite(save_path, face_img)
    
    # Extract face encodings
    encode = face_recognition.face_encodings(face_img)[0]
    return encode

# Load train and validation datasets
X_train, y_train = load_dataset(train_path)
X_val, y_val = load_dataset(val_path)

print("Training data shape: ", X_train.shape, y_train.shape)
print("Validation data shape: ", X_val.shape, y_val.shape)

# Train SVM model
svc_model = svm.SVC(gamma='scale')
svc_model.fit(X_train, y_train)

# Train accuracy
train_pred = svc_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("SVM Training Accuracy: ", train_acc)

# Validation accuracy
val_pred = svc_model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("SVM Validation Accuracy: ", val_acc)

# Classification report
print("SVM Classification Report:\n", classification_report(y_val, val_pred))

# Save SVM model
model_name = 'svm-{}.model'.format(str(int(val_acc * 100)))
pickle.dump(svc_model, open(model_name, 'wb'))

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# KNN train accuracy
train_pred = knn_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("KNN Training Accuracy: ", train_acc)

# KNN validation accuracy
val_pred = knn_model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("KNN Validation Accuracy: ", val_acc)

# Classification report
print("KNN Classification Report:\n", classification_report(y_val, val_pred))

# Save KNN model
model_name = 'knn-{}.model'.format(str(int(val_acc * 100)))
pickle.dump(knn_model, open(model_name, 'wb'))

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

# RF train accuracy
train_pred = rf_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("RF Training Accuracy: ", train_acc)

# RF validation accuracy
val_pred = rf_model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("RF Validation Accuracy: ", val_acc)

# Classification report
print("RF Classification Report:\n", classification_report(y_val, val_pred))

# Save RF model
model_name = 'rf-{}.model'.format(str(int(val_acc * 100)))
pickle.dump(rf_model, open(model_name, 'wb'))
