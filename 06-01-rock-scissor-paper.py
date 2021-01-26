import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        # Data Augmentation from below
        rotation_range=0.4,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        subset='validation'
    )


    model = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    return model, train_generator, validation_generator


model, train_generator, validation_generator = solution_model()

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, model_checkpoint]
)

model.save('./rps_model.h5')

fig, axes = plt.subplots(1, 2, figsize=(15,4))

axes[0].plot(history.history['val_loss'], color='b', label="val loss")
axes[0].set_title("Vali Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history['loss'], color='green', label="loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.show()

