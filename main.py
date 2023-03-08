import keras
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/uzytkownik/Downloads/train.csv')
print(data)
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = keras.Sequential([
    Dense(64, activation='sigmoid', input_shape=(784,)),
    Dense(32, 'sigmoid'),
    Dense(16, 'sigmoid'),
    Dense(10, 'sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=1,
)
history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
plt.show()