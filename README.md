# 🏠 House Prices: Prediction & Deployment

This project contains a machine learning model for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and a web application to showcase the results, deployed via GitHub Pages.

## 🚀 Live Demo
Visit the live application here: **[https://your-username.github.io/Housing_prices/](https://your-username.github.io/Housing_prices/)**
*(Replace with your actual GitHub Pages URL once deployed)*

## ✨ Features
- **Advanced Stacking Ensemble**: Combines XGBoost, Gradient Boosting, Ridge, and Lasso regression for high-accuracy predictions.
- **Serverless Web App**: Built with **Streamlit** and hosted on **GitHub Pages** using **Stlite** (WebAssembly).
- **No Backend Required**: The model runs entirely in your web browser.

## 🛠️ Project Structure
- `app.py`: The Streamlit web application.
- `index.html`: Entry point for Stlite (WebAssembly) hosting.
- `train_and_save.py`: Script to train the model and save artifacts.
- `static/`: Contains serialized model and preprocessing objects (`.joblib`).
- `home-data-for-ml-course/`: Training and test data files.
- `housingprice-test .ipynb`: Original exploration and modeling notebook.

## 💻 Local Setup

### Running the Web App locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

### Training the Model
If you want to retrain the model:
```bash
python train_and_save.py
```

## 🌐 Deployment to GitHub Pages
The project is configured to run as a static site. To deploy:
1. Push your code to a GitHub repository.
2. Go to **Settings > Pages**.
3. Set the source to **Deploy from a branch** and select `main` (root).
4. Your site will be live within minutes!

## 📜 Acknowledgments
- Data provided by the Kaggle Housing Prices competition.
- WebAssembly support provided by [Stlite](https://github.com/whitphx/stlite).
