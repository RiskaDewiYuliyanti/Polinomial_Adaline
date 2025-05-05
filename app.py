import streamlit as st
import numpy as np
import pickle
import os

# --- Definisi Ulang Class Adaline (WAJIB ADA) ---
class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def activation(self, x):
        return x  # Linear activation untuk Adaline

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_predicted = self.predict(X)
            errors = y - y_predicted

            self.weights += self.learning_rate * np.dot(X.T, errors) / n_samples
            self.bias += self.learning_rate * errors.mean()

            mse = np.mean(errors ** 2)
            self.loss_history.append(mse)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - MSE: {mse:.6f}")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# --- Load Model ---
MODEL_PATH = "adaline_polynomial_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Streamlit Layout ---
st.set_page_config(page_title="Solusi Akar Polinomial | Adaline AI", layout="wide")

st.title("🔢 Solusi Akar Persamaan Polinomial Derajat Tinggi")
st.subheader("Menggunakan Artificial Neural Network Model Adaline")
st.markdown("---")

# Sidebar Input
st.sidebar.header("Input Parameter")
st.sidebar.markdown("**Input Polinomial**")

degree = st.sidebar.number_input("Masukkan Derajat Polinomial (n):", min_value=1, value=2, step=1)

coefficients = []
for i in range(degree + 1):
    coef = st.sidebar.number_input(f"Koefisien x^{degree-i}:", format="%.5f", key=f"coef_{i}")
    coefficients.append(coef)

input_x = st.sidebar.number_input("Masukkan nilai x untuk prediksi:", format="%.4f")

# Action Button
if st.sidebar.button("Prediksi"):
    st.markdown("## 📈 Hasil Prediksi")

    # Buat prediksi
    poly_value = np.polyval(coefficients, input_x)
    st.write(f"Nilai polinomial di x={input_x} adalah: **{poly_value:.6f}**")

    # Prediksi dengan model Adaline
    x_input = np.array([[input_x]])
    adaline_prediction = model.predict(x_input)
    st.success(f"Hasil prediksi model Adaline di x={input_x}: **{adaline_prediction[0]:.6f}**")

    st.info("Note: Model Adaline ini hasil training dari dataset polinomial contoh. Untuk polinomial baru, perlu re-training.")

# Visualisasi Persamaan
st.markdown("## 📄 Persamaan Polinomial")
equation = " + ".join([f"{c:.2f}x^{degree-i}" for i, c in enumerate(coefficients)])
equation = equation.replace('x^0', '')
st.latex(equation)

# Tambahan (optional): Gambar Diagram
if os.path.exists("assets/flowmap.png"):
    st.markdown("---")
    st.markdown("## 🛠️ Diagram Flowmap Sistem")
    st.image("assets/flowmap.png", caption="Flowmap Sistem Solusi Polinomial", use_column_width=True)

# Footer
st.markdown("---")
st.caption("© 2025 Solusi Polinomial Adaline | Powered by Streamlit & AI")
st.markdown("**Disclaimer:** Model ini hanya untuk tujuan edukasi. Hasil prediksi tidak dijamin akurat untuk semua polinomial.")
