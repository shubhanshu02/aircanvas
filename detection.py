import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CharacterDetector:
    def __init__(self, train=False, loadFile=""):
        self.createModel()

        # Dictionary for getting characters from index values...
        self.word_dict = {}
        for i in range(0, 26):
            self.word_dict[i] = chr(65 + i)

        if train:
            self.dataset()

        if loadFile:
            if self.loadModel(loadFile):
                print("Model loaded successfully...")
            else:
                import sys

                print("Unable to Load model...")
                sys.exit()

    def loadModel(self, loadFile):
        if self.model:
            self.model.load_weights(loadFile)
            return True
        return False

    def dataset(self):
        # Read the data...
        data = pd.read_csv(r"A_Z Handwritten Data.csv").astype("float32")
        X = data.drop("0", axis=1)
        y = data["0"]

        # Split data the X - Our data , and y - the prdict label
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
        test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

        # Reshaping the data in csv file so that it can be displayed as an img...
        print("Train data shape: ", train_x.shape)
        print("Test data shape: ", test_x.shape)

        # Plotting the number of alphabets in the dataset...

        train_yint = np.int0(y)
        count = np.zeros(26, dtype="int")
        for i in train_yint:
            count[i] += 1

        alphabets = []
        for i in self.word_dict.values():
            alphabets.append(i)

        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.barh(alphabets, count)

        plt.xlabel("Number of elements ")
        plt.ylabel("Alphabets")
        plt.grid()
        plt.show()

        # Shuffling the data ...
        shuff = shuffle(train_x[:100])

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        axes = ax.flatten()
        for i in range(9):
            axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap="Greys")
        plt.show()
        """

        # Reshaping the training & test dataset so that it can be put in the model...
        self.trainX = train_x.reshape(
            train_x.shape[0], train_x.shape[1], train_x.shape[2], 1
        )
        print("New shape of train data: ", self.trainX.shape)

        self.testX = test_x.reshape(
            test_x.shape[0], test_x.shape[1], test_x.shape[2], 1
        )
        print("New shape of train data: ", self.testX.shape)

        # Converting the labels to categorical values...
        self.trainY = to_categorical(train_y, num_classes=26, dtype="int")
        print("New shape of train labels: ", self.trainY.shape)

        self.testY = to_categorical(test_y, num_classes=26, dtype="int")
        print("New shape of test labels: ", self.testY.shape)

    def createModel(self):
        # CNN model...
        model = Sequential(
            [
                Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=(28, 28, 1),
                ),
                MaxPool2D(pool_size=(2, 2), strides=2),
                Dropout(0.3),
                Conv2D(
                    filters=64, kernel_size=(3, 3), activation="relu", padding="same"
                ),
                MaxPool2D(pool_size=(2, 2), strides=2),
                Dropout(0.3),
                Conv2D(
                    filters=128, kernel_size=(3, 3), activation="relu", padding="valid"
                ),
                MaxPool2D(pool_size=(2, 2), strides=2),
                Dropout(0.3),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(128, activation="relu"),
                Dense(26, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        self.model = model
        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001
        )
        self.early_stop = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=2, verbose=0, mode="auto"
        )

    def train(self):
        if self.trainX is None:
            print("Dataset not initialized!")
            return
        history = self.model.fit(
            self.trainX,
            self.trainY,
            epochs=1,
            callbacks=[self.reduce_lr, self.early_stop],
            validation_data=(self.testX, self.testY),
        )
        self.model.save(r"model_hand.h5")

        # Displaying the accuracies & losses for train & validation set...
        print("The validation accuracy is :", history.history["val_accuracy"])
        print("The training accuracy is :", history.history["accuracy"])
        print("The validation loss is :", history.history["val_loss"])
        print("The training loss is :", history.history["loss"])

    def test(self):
        # Making model predictions...
        pred = self.model.predict(self.testX[:9])
        print(self.testX.shape)

        # Displaying some of the test imgs & their predicted labels...
        fig, axes = plt.subplots(3, 3, figsize=(8, 9))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            img = np.reshape(self.testX[i], (28, 28))
            ax.imshow(img, cmap="Greys")
            pred = self.word_dict[np.argmax(self.testY[i])]
            ax.set_title("Prediction: " + pred)
            ax.grid()

    def predict(self, img):
        if type(img) == str:
            # Prediction on external img...
            img = cv2.imread(img)

        img_copy = img.copy()
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 440))

        img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        cv2.imshow("Recognised", img_final)
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        img_pred = self.word_dict[np.argmax(self.model.predict(img_final))]
        cv2.putText(
            img,
            "Prediction: " + img_pred,
            (20, 410),
            cv2.FONT_HERSHEY_DUPLEX,
            1.3,
            color=(255, 0, 30),
        )
        cv2.imshow("Recognised Character", img)
        if __name__ == "__main__":
            while 1:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()
        return img_pred


# det = CharacterDetector(train=True, loadFile="model.h5")
# det.predict("c.jpeg")
