import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# --- Definisi Class Adaline ---
class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000, initial_weights=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = initial_weights
        self.bias = 0
        self.loss_history = []

    def activation(self, x):
        return x

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)

    def train(self, X, y):
        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            errors = y - y_pred

            self.weights += self.learning_rate * np.dot(X.T, errors) / n_samples
            self.bias += self.learning_rate * errors.mean()

            mse = np.mean(errors ** 2)
            self.loss_history.append(mse)

# --- UI Streamlit ---
st.title("Simulasi Adaline - Training Polinomial")

# Input parameter dari user
learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
epochs = st.number_input("Jumlah Iterasi (Epoch)", min_value=100, max_value=10000, value=1000, step=100)
initial_weight = st.number_input("Bobot Awal (W)", value=0.0)

if st.button("Adaline (Proses)"):

    # Memuat model yang telah disimpan
    model = joblib.load('model.pkl')

    # Tampilkan info hasil
    st.write(f"Bobot akhir: {model.weights}")
    st.write(f"Bias akhir: {model.bias}")
    st.write(f"Galat akhir (MSE): {model.loss_history[-1]:.4f}")

    # Grafik MSE
    st.subheader("Grafik MSE per Epoch")
    fig, ax = plt.subplots()
    ax.plot(model.loss_history, color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Perkembangan MSE Selama Training")
    st.pyplot(fig)
