from sklearn.base import BaseEstimator, TransformerMixin
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.decomposition import PCA
from keras.layers import Input, Dense
from keras.models import Model


class AutoEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, layers=(64, 10, 64)):
        self.layers = layers
        self.encoder = None

    def fit(self, X, y=None):
        input_dims = Input(shape=(180,))
        encoded = Dense(64, activation='relu')(input_dims)
        encoded = Dense(10, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(180, activation='sigmoid')(decoded)

        autoencoder = Model(input_dims, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X, X,
                        epochs=1,
                        batch_size=50,
                        shuffle=True, )
        self.encoder = Model(input_dims, encoded)
        return self

    def transform(self, X, y=None):
        return self.encoder.predict(X)
