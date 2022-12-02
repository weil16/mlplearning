from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Activation, Dropout
#from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import SGD
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar100


(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape


X_train[0, :].shape # image shape



to_categorical(y_train)[0, :].shape # number of categories


def visualize_random_images(images):
    plt.figure(figsize=(6, 6))
    for ind, img in enumerate(images[:9, :]):
        plt.subplot(int("33%d" % (ind + 1)))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
visualize_random_images(X_train)


model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))




oh = OneHotEncoder(sparse=False)
oh.fit(y_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train/255., oh.transform(y_train), epochs=10, batch_size=32,
          validation_data=(X_test/255., oh.transform(y_test)))

model.summary()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()


