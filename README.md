# MNIST Digit Recognizer

Three models (SVM, MLP and CNN) are trained on the MNIST handwritten digits, compared, and used in a small web app:
you draw a digit, pick a model, and get a prediction.

| Model | Test accuracy |
|---|---|
| SVM (classical machine learning) | 98.51% |
| MLP (neural network) | 98.36% |
| CNN (deep learning) | **99.02%** |

The full write-up is in [reports/MNIST_Model_Report.pdf](reports/MNIST_Model_Report.pdf).

## Run the application

You need Python 3.11 or newer. Open a terminal in this folder and run:

```
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/app.py
```

Then open **http://127.0.0.1:5000** in your browser:

1. Draw one large, thick digit on the black canvas.
2. Choose a model (SVM, MLP or CNN) and press **Predict**.
3. Click **Model comparison** at the top to see how the three models compare.

Press `Ctrl+C` in the terminal to stop the app. The trained models are already included in `app/models/`, so you do not
need to train anything to run the app.

## Train the models again (optional)

Open the notebooks in the `notebooks/` folder and run all cells, in this order:

1. `01_svm.ipynb`: the SVM (the slowest one, it takes a few minutes)
2. `02_mlp.ipynb`: the MLP
3. `03_cnn.ipynb`: the CNN
4. `04_comparison.ipynb`: the comparison table and chart (run it after the other three)

The first notebook downloads MNIST into `data/`. Each notebook saves its model to `app/models/` and its results to
`artifacts/`. Restart the app afterwards so it loads the new models.

## Project structure

```
app/            the web application
  app.py          the web server
  inference.py    prepares a drawing and runs the chosen model
  model_utils.py  the MLP and CNN definitions (must match the notebooks)
  image_utils.py  image constants and reshaping helpers
  templates/      the two web pages
  static/         the style sheet
  models/         the three trained models
  charts/         the comparison table and charts shown on the website
notebooks/      training code for each model and the comparison
artifacts/      results saved by the notebooks
reports/        the report (PDF)
requirements.txt
```
