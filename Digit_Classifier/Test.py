#!/usr/bin/env python3
"""
Draw & Compare Digits — Interactive MNIST-style tester

Features
- Canvas to draw digits with mouse
- Choose between two built-in models (LogReg & RandomForest) trained on sklearn 8x8 digits
- (Optional) If TensorFlow + a Keras MNIST model (.h5) is available, add as a third model
- Predict on every submit; you provide the true label via 0–9 buttons
- Tracks per-model accuracy over time and plots it live
- Shows model probabilities for the current drawing

Dependencies
- Python 3.9+ recommended
- tkinter (usually bundled)
- pillow (PIL): pip install pillow
- numpy: pip install numpy
- scikit-learn: pip install scikit-learn
- matplotlib: pip install matplotlib
- (Optional) tensorflow: pip install tensorflow

Run
    python draw_and_compare_digits.py
"""

import io
import sys
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# matplotlib embedding
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# scikit-learn models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Try to import tensorflow lazily (optional third model support)
try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------------------
# Utility: Model abstractions
# ---------------------------

class BaseDigitModel:
    """Abstract base class for digit models."""
    name: str = "Base"

    def predict_proba_from_image(self, img: Image.Image) -> np.ndarray:
        """Return probability distribution over classes 0..9 for input PIL Image.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class SklearnDigits8x8Model(BaseDigitModel):
    """Pipeline that expects an 8x8 grayscale digit (sklearn digits format)."""

    def __init__(self, name: str, classifier):
        self.name = name
        # We'll build a pipeline: [8x8 -> flatten(64)] + StandardScaler + classifier
        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", classifier)
        ])
        self._fit_on_sklearn_digits()

    def _fit_on_sklearn_digits(self):
        digits = load_digits()
        X, y = digits.data, digits.target  # X is already 64-length flattened 8x8
        # Split and train for speed; accuracy is decent for this app
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.pipeline.predict(X_test))
        print(f"[{self.name}] test accuracy on sklearn digits: {acc:.3f}")

    @staticmethod
    def _pil_to_8x8_vector(img: Image.Image) -> np.ndarray:
        """Convert a drawn image (white on black) to an 8x8 vector like sklearn digits.
        sklearn digits are 0..16 intensity. We'll downsample and scale accordingly.
        """
        # Convert to grayscale, invert so digit is dark on light if needed
        g = img.convert("L")
        # Downscale to 8x8 using antialias
        g8 = g.resize((8, 8), Image.LANCZOS)
        # Normalize to 0..16 like sklearn digits dataset
        arr = np.asarray(g8).astype(np.float32)
        # Images we draw are white strokes on black background; invert for sklearn-like
        arr = 255.0 - arr
        arr = (arr / 255.0) * 16.0
        return arr.reshape(1, -1)

    def predict_proba_from_image(self, img: Image.Image) -> np.ndarray:
        X = self._pil_to_8x8_vector(img)
        # Some sklearn classifiers may not implement predict_proba; handle gracefully
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba(X)[0]
        else:
            # Use decision_function if available, else pseudo-probabilities
            if hasattr(clf, "decision_function"):
                scores = self.pipeline.decision_function(X)[0]
            else:
                # Fallback: predict label and make a one-hot-ish vector
                pred = int(self.pipeline.predict(X)[0])
                scores = np.full(10, -1e9, dtype=np.float32)
                scores[pred] = 0.0
            # Softmax
            e = np.exp(scores - np.max(scores))
            probs = e / e.sum()
        return probs


class KerasMNIST28x28Model(BaseDigitModel):
    """Optional: Keras model that expects (28,28,1) MNIST-style input.
    Only enabled if TensorFlow is installed and a .h5 file is loaded.
    """

    def __init__(self, h5_path: str):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available.")
        self.name = f"KerasMNIST[{h5_path}]"
        self.model = tf.keras.models.load_model(h5_path)
        # Verify 10-class softmax
        if self.model.output_shape[-1] != 10:
            raise ValueError("Loaded Keras model does not output 10 classes.")

    @staticmethod
    def _pil_to_28x28_batch(img: Image.Image) -> np.ndarray:
        g = img.convert("L").resize((28, 28), Image.LANCZOS)
        # Drawn strokes are white on black; MNIST is white digit on black background too.
        arr = np.asarray(g).astype(np.float32) / 255.0
        # Ensure the digit is bright; optional normalize
        arr = arr[..., None]  # (28,28,1)
        return arr[None, ...]  # (1,28,28,1)

    def predict_proba_from_image(self, img: Image.Image) -> np.ndarray:
        x = self._pil_to_28x28_batch(img)
        preds = self.model.predict(x, verbose=0)[0]
        return preds.astype(np.float32)


# ---------------------------
# App State & Metrics
# ---------------------------

@dataclass
class ModelStats:
    name: str
    history_true: List[int] = field(default_factory=list)
    history_pred: List[int] = field(default_factory=list)
    history_acc: List[float] = field(default_factory=list)  # rolling accuracy after each submission

    def update(self, y_true: int, y_pred: int):
        self.history_true.append(y_true)
        self.history_pred.append(y_pred)
        acc = (np.array(self.history_true) == np.array(self.history_pred)).mean()
        self.history_acc.append(float(acc))

    def current_accuracy(self) -> float:
        if not self.history_acc:
            return 0.0
        return self.history_acc[-1]


# ---------------------------
# Tkinter GUI
# ---------------------------

class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Draw & Compare Digits")
        self.root.geometry("1100x720")

        # Drawing canvas state
        self.canvas_size = 280
        self.brush_size = 18
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Create PIL image backing store (black background)
        self.pil_img = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        # Models
        self.models: Dict[str, BaseDigitModel] = {}
        self.stats: Dict[str, ModelStats] = {}
        self.current_model_name: Optional[str] = None

        # Build UI
        self._build_left_panel()
        self._build_right_panel()
        self._load_default_models()

        self._bind_events()
        self._refresh_prediction()

    # UI layout
    def _build_left_panel(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        # Canvas
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(pady=(0, 8))

        # Buttons row
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Invert", command=self.invert_canvas).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Erode", command=self.erode).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Dilate", command=self.dilate).pack(side=tk.LEFT, padx=4)

        # Brush size
        bs_frame = ttk.Frame(left)
        bs_frame.pack(fill=tk.X, pady=6)
        ttk.Label(bs_frame, text="Brush size").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=self.brush_size)
        bs = ttk.Scale(bs_frame, from_=4, to=36, orient=tk.HORIZONTAL, command=self._on_brush_change)
        bs.set(self.brush_size)
        bs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # True label selector
        ttk.Label(left, text="Select the TRUE label, then press Submit:").pack(pady=(12, 4))
        digits_frame = ttk.Frame(left)
        digits_frame.pack()
        self.true_label_var = tk.IntVar(value=0)
        for d in range(10):
            b = ttk.Radiobutton(digits_frame, text=str(d), value=d, variable=self.true_label_var)
            b.grid(row=d//5, column=d%5, padx=6, pady=4)

        submit_row = ttk.Frame(left)
        submit_row.pack(pady=(8, 0))
        ttk.Button(submit_row, text="Submit (score it)", command=self.submit_and_score).pack(side=tk.LEFT, padx=4)
        ttk.Button(submit_row, text="Predict only", command=self._refresh_prediction).pack(side=tk.LEFT, padx=4)

    def _build_right_panel(self):
        right = ttk.Frame(self.root, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Model selector
        model_row = ttk.Frame(right)
        model_row.pack(fill=tk.X)
        ttk.Label(model_row, text="Active model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT, padx=8, pady=4)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_changed())

        ttk.Button(model_row, text="Add Keras .h5 (optional)", command=self.add_keras_model).pack(side=tk.LEFT, padx=4)

        # Prediction readout
        pred_box = ttk.LabelFrame(right, text="Prediction")
        pred_box.pack(fill=tk.X, pady=(10, 6))
        self.pred_label = ttk.Label(pred_box, text="Top-1: —", font=("TkDefaultFont", 12, "bold"))
        self.pred_label.pack(anchor="w", padx=6, pady=4)
        self.prob_text = tk.Text(pred_box, height=6, width=60)
        self.prob_text.pack(fill=tk.X, padx=6, pady=(0,6))

        # Accuracy chart
        chart_box = ttk.LabelFrame(right, text="Session Accuracy (per model)")
        chart_box.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Submissions")
        self.ax.set_ylabel("Accuracy")
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, alpha=0.3)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=chart_box)
        self.canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ---------------------------
    # Drawing operations
    # ---------------------------
    def _on_press(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        self._draw_point(event.x, event.y)

    def _on_move(self, event):
        if not self.drawing:
            return
        self._draw_line(self.last_x, self.last_y, event.x, event.y)
        self.last_x, self.last_y = event.x, event.y

    def _on_release(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None
        self._refresh_prediction()

    def _on_brush_change(self, value):
        try:
            self.brush_size = int(float(value))
        except Exception:
            pass

    def _draw_point(self, x, y):
        r = self.brush_size // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.pil_draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

    def _draw_line(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2, fill="white", width=self.brush_size, capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line((x1, y1, x2, y2), fill=255, width=self.brush_size)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_img.paste(0)
        self._refresh_prediction()

    def invert_canvas(self):
        self.pil_img = ImageOps.invert(self.pil_img.convert("L"))
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        # redraw to Tk canvas
        self._redraw_tk_from_pil()
        self._refresh_prediction()

    def erode(self):
        self._morph_kernel(op="erode")

    def dilate(self):
        self._morph_kernel(op="dilate")

    def _morph_kernel(self, op="erode", k=3):
        """Simple morphological ops via min/max filter."""
        if op == "erode":
            self.pil_img = self.pil_img.filter(ImageFilter.MinFilter(k))
        else:
            self.pil_img = self.pil_img.filter(ImageFilter.MaxFilter(k))
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self._redraw_tk_from_pil()
        self._refresh_prediction()

    def _redraw_tk_from_pil(self):
        # Render PIL image to TK Canvas by creating an image background
        # We'll clear and paste as rectangles for simplicity
        self.canvas.delete("all")
        # Convert to a PhotoImage via Tk's supported format
        tmp = self.pil_img.resize((self.canvas_size, self.canvas_size), Image.NEAREST)
        # Draw white pixels as small rects (avoid heavy PhotoImage conversions)
        arr = np.array(tmp)
        ys, xs = np.where(arr > 0)
        for y, x in zip(ys, xs):
            self.canvas.create_rectangle(x, y, x+1, y+1, outline="", fill="white")

    # ---------------------------
    # Modeling
    # ---------------------------
    def _load_default_models(self):
        # Two sklearn models on 8x8 digits
        logreg = SklearnDigits8x8Model("LogisticRegression(8x8)", LogisticRegression(max_iter=2000))
        rf = SklearnDigits8x8Model("RandomForest(8x8)", RandomForestClassifier(n_estimators=200, random_state=42))

        self._register_model(logreg)
        self._register_model(rf)

        self.model_combo["values"] = list(self.models.keys())
        if self.model_combo["values"]:
            self.model_combo.current(0)
            self.current_model_name = self.model_combo.get()

    def _register_model(self, model: BaseDigitModel):
        self.models[model.name] = model
        self.stats[model.name] = ModelStats(name=model.name)

    def add_keras_model(self):
        if not TF_AVAILABLE:
            messagebox.showwarning("TensorFlow not available", "TensorFlow is not installed; cannot add Keras model.")
            return
        path = filedialog.askopenfilename(
            title="Select Keras (.h5) model",
            filetypes=[("Keras model", "*.h5 *.keras"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            km = KerasMNIST28x28Model(path)
        except Exception as e:
            messagebox.showerror("Failed to load model", f"{e}")
            return
        self._register_model(km)
        self.model_combo["values"] = list(self.models.keys())
        self.model_combo.set(km.name)
        self.current_model_name = km.name
        self._refresh_prediction()

    def _on_model_changed(self):
        self.current_model_name = self.model_combo.get()
        self._refresh_prediction()

    def _get_active_model(self) -> Optional[BaseDigitModel]:
        if not self.current_model_name:
            return None
        return self.models.get(self.current_model_name, None)

    def _refresh_prediction(self):
        model = self._get_active_model()
        if model is None:
            self.pred_label.config(text="Top-1: —")
            self.prob_text.delete("1.0", tk.END)
            return
        probs = model.predict_proba_from_image(self.pil_img)
        pred = int(np.argmax(probs))
        top1 = float(probs[pred])
        self.pred_label.config(text=f"Top-1: {pred}  (p={top1:.3f})  —  Model: {model.name}")
        # Show full distribution
        lines = [f"{d}: {probs[d]:.3f}" for d in range(10)]
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert(tk.END, "\n".join(lines))

    def submit_and_score(self):
        true_label = int(self.true_label_var.get())
        model = self._get_active_model()
        if model is None:
            messagebox.showwarning("No model", "Select a model first.")
            return
        probs = model.predict_proba_from_image(self.pil_img)
        pred = int(np.argmax(probs))
        self.stats[model.name].update(true_label, pred)
        self._update_chart()
        # Visual feedback
        corr = "✔️ correct" if pred == true_label else "❌ wrong"
        messagebox.showinfo("Result", f"Pred: {pred}  (p={probs[pred]:.3f}) — {corr}")

    def _update_chart(self):
        self.ax.clear()
        self.ax.set_xlabel("Submissions")
        self.ax.set_ylabel("Accuracy")
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, alpha=0.3)

        for name, st in self.stats.items():
            if st.history_acc:
                xs = np.arange(1, len(st.history_acc) + 1)
                self.ax.plot(xs, st.history_acc, label=f"{name} (n={len(xs)})")
        self.ax.legend(loc="lower right", framealpha=0.8)
        self.canvas_mpl.draw_idle()


def main():
    root = tk.Tk()
    # Make ttk look a bit nicer
    try:
        from tkinter import font
        default = font.nametofont("TkDefaultFont")
        default.configure(size=11)
    except Exception:
        pass

    app = DigitApp(root)
    root.mainloop()


if __name__ == "__main__":
    # Optional: speed up MacOS Quartz font rendering warnings
    try:
        import warnings
        warnings.filterwarnings("ignore")
    except Exception:
        pass
    main()
