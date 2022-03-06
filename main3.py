from gc import callbacks
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = './Dataset'
# NSFW - 1, SFW - 0
label = [0, 1]
size = (100, 100)

X_train, X_test, Y_train, Y_test = [], [], [], []

for i in os.listdir(path):
    if i == 'test':
        for j in os.listdir(os.path.join(f'{path}\{i}', i)):
            try:
                img = cv2.imread(os.path.join(f'{path}\{i}\{i}', j))
                img = cv2.resize(img, size)
                X_test.append(img)
                if j[0] == 'N':
                    Y_test.append(label[1])
                else:
                    Y_test.append(label[0])
            except:
                print('test error')
    else:
        for j in os.listdir(os.path.join(f'{path}\{i}', i)):
            for k in os.listdir(os.path.join(f'{path}\{i}\{i}', j)):
                try:
                    img = cv2.imread(os.path.join(f'{path}\{i}\{i}\{j}', k))
                    img = cv2.resize(img, size)
                    X_train.append(img)
                    if j[0] == 'N':
                        Y_train.append(label[1])
                    else:
                        Y_train.append(label[0])
                except:
                    print('train error')
                
print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = X_train/255.0
X_test = X_test/255.0
# X_train = X_train.reshape((len(X_train), 50*50*3))
# X_test = X_test.reshape((len(X_test), 50*50*3))

print(X_train[0].shape)
print(X_train.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
checkpoint_filepath1 = 'lmodel/val_accuracy'
checkpoint_filepath2 = 'lmodel/accuracy'

model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2,
    monitor='accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), batch_size=100, callbacks=[model_checkpoint_callback1, model_checkpoint_callback2])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print(test_acc)