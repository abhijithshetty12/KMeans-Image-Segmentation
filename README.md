# ColorLab: K-Means Image Segmentation

A local web application for a Machine Learning Module 4 Innovative Exam demo.

## Quick start

Install Python 3.10+ and run these commands from this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The local website normally opens at `http://localhost:8501`.

## Features
- Upload JPG, PNG, or WebP photos.
- Select K (2–12) or choose an automatic K *heuristic*.
- Show original vs. K-Means segmented image.
- Isolate and download each RGB color group as a transparent PNG.
- Download all results as a ZIP and inspect cluster proportions.
- Optionally use MiniBatchKMeans for faster processing.

## Important for your research-paper implementation assessment

This app implements standard RGB K-Means with an optional silhouette-based K estimation.
It **does not implement** the specific PCA-based automatic K method described by
Sabha, Mousa & Maree (2026). You must implement and validate that paper's exact
algorithm separately before claiming this is a reproduction of that paper.

K-Means segments by color; it does not understand object categories. A bird and
background sharing colors may appear in the same cluster.

Images are resized to fit 900x900 for responsiveness. Model fitting may use up to
100,000 sampled pixels while predicting cluster labels for all displayed pixels.
