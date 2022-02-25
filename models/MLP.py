import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from utils import return_metrics

def train_MLP(X_train, y_train, X_validation, y_validation):
    model = Sequential()
    model.add(Dense(160, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto')

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy']
                  )

    model.fit(X_train, y_train,
              verbose=1,
              epochs=200,
              batch_size=100,
              callbacks=[early_stop_callback],
              validation_data=(X_validation, y_validation))

    y_pred = model.predict(X_validation)
    print(f' Validation F1-score and log-loss: {return_metrics(y_validation, y_pred)}')
    return model