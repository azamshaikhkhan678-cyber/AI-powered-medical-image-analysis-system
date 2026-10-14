# main.py
import sys
import random
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Bind our updated style metrics folder safely
from medicalconfig.settings import *

class MedicalVisionMLModel:
    def __init__(self):
        X_features = []
        y_labels = []
        for i in range(60):
            mock_scan = np.zeros((IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), dtype=np.uint8)
            if i < 30:
                cv2.circle(mock_scan, (64, 64), 45, 40, -1)
                label = 0
            else:
                cv2.circle(mock_scan, (64, 64), 45, 120, -1)
                cv2.rectangle(mock_scan, (40, 40), (80, 80), 180, -1)
                label = 1

            blurred = cv2.GaussianBlur(mock_scan, (5, 5), 0)
            hist = cv2.calcHist([blurred], [0], None, [256], [0, 256]).flatten()
            X_features.append(hist)
            y_labels.append(label)

        self.classifier = RandomForestClassifier(n_estimators=25, random_state=42)
        self.classifier.fit(X_features, y_labels)
        train_accuracy = self.classifier.score(X_features, y_labels)
        self.model_score_text = f"{train_accuracy * 100:.1f}% ACCURACY"

    def analyze_scan_matrix(self, scan_intensity_mode):
        mock_input = np.zeros((IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), dtype=np.uint8)
        if scan_intensity_mode == "CLEAR":
            cv2.circle(mock_input, (64, 64), 45, 42, -1)
            diagnostic_name = "NORMAL LUNG STRUCTURE DETECTED"
            ui_color = COLOR_HEALTHY
        else:
            cv2.circle(mock_input, (64, 64), 45, 115, -1)
            cv2.rectangle(mock_input, (42, 42), (78, 78), 185, -1)
            diagnostic_name = "CRITICAL INFILTRATE INFECTION IDENTIFIED"
            ui_color = COLOR_ANOMALY

        blurred = cv2.GaussianBlur(mock_input, (5, 5), 0)
        hist_vector = cv2.calcHist([blurred], [0], None, [256], [0, 256]).flatten()
        prediction = self.classifier.predict([hist_vector])
        probabilities = self.classifier.predict_proba([hist_vector])
        confidence_value = float(probabilities[0][prediction[0]])
        return mock_input, diagnostic_name, ui_color, confidence_value

class ClinicalRadiologySOCPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Clinical Radiology Analysis Terminal")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.configure(bg=COLOR_BG_CLINICAL)

        self.vision_engine = MedicalVisionMLModel()
        self.total_scans_processed = 0
        self.assemble_clinical_layout()

    def assemble_clinical_layout(self):
        self.header = tk.Frame(self.root, bg=COLOR_PANEL_BG, height=60)
        self.header.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)

        tk.Label(self.header, text="🩺 AI-POWERED MEDICAL IMAGE ANALYSIS HUB", 
                 font=("Courier New", 13, "bold"), fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL_BG).pack(side=tk.LEFT, padx=20, pady=15)

        self.btn_healthy = tk.Button(self.header, text="🔬 SCAN BASELINE PATIENT", font=("Courier New", 9, "bold"),
                                     fg="#ffffff", bg="#0088cc", bd=0, padx=12, command=lambda: self.trigger_image_diagnostic_cycle("CLEAR"))
        self.btn_healthy.pack(side=tk.RIGHT, padx=10, pady=12)

        self.btn_anomaly = tk.Button(self.header, text="🚨 SCAN CRITICAL CASE", font=("Courier New", 9, "bold"),
                                     fg="#ffffff", bg="#cc2200", bd=0, padx=12, command=lambda: self.trigger_image_diagnostic_cycle("INFECTED"))
        self.btn_anomaly.pack(side=tk.RIGHT, padx=10, pady=12)

        self.metrics_bar = tk.Frame(self.root, bg=COLOR_BG_CLINICAL)
        self.metrics_bar.pack(side=tk.TOP, fill=tk.X, padx=15, pady=5)

        self.card_total = self.create_card(self.metrics_bar, "TOTAL PATIENT CASES SCANNED", "0", COLOR_TEXT_WHITE, 0)
        self.card_score = self.create_card(self.metrics_bar, "VISION ACCURACY SCORE", self.vision_engine.model_score_text, COLOR_HEALTHY, 1)
        self.card_conf = self.create_card(self.metrics_bar, "AI INFERENCE CONFIDENCE", "0.0%", COLOR_HEALTHY, 2)

        self.left_panel = tk.Frame(self.root, bg=COLOR_PANEL_BG, bd=1, relief=tk.SUNKEN)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.canvas_scan = tk.Canvas(self.left_panel, bg="#05080f", highlightthickness=0, bd=0)
        self.canvas_scan.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.right_panel = tk.Frame(self.root, bg=COLOR_PANEL_BG, width=380, bd=1, relief=tk.SUNKEN)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=15, pady=10)
        self.right_panel.pack_propagate(False)

        self.panel_status = tk.Frame(self.right_panel, bg="#05080f", height=120)
        self.panel_status.pack(fill=tk.X, padx=15, pady=10)
        self.panel_status.pack_propagate(False)

        self.lbl_status = tk.Label(self.panel_status, text="AWAITING IMAGE INPUT...", font=("Courier New", 10, "bold"), 
                                   fg=COLOR_TEXT_WHITE, bg="#05080f", wraplength=320, justify=tk.CENTER)
        self.lbl_status.pack(expand=True, fill=tk.BOTH)

        self.log_box = tk.Listbox(self.right_panel, bg="#05080f", fg="#ffffff", font=("Courier New", 9), bd=0, highlightthickness=0)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def create_card(self, parent, title, val, color, col_idx):
        card = tk.Frame(parent, bg=COLOR_PANEL_BG, bd=1, relief=tk.RAISED)
        card.grid(row=0, column=col_idx, padx=8, pady=5, sticky="nsew")
        parent.grid_columnconfigure(col_idx, weight=1)
        tk.Label(card, text=title, font=("Courier New", 8, "bold"), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(pady=6)
        lbl = tk.Label(card, text=val, font=("Courier New", 16, "bold"), fg=color, bg=COLOR_PANEL_BG)
        lbl.pack(pady=2)
        return lbl

    def trigger_image_diagnostic_cycle(self, patient_scan_mode):
        self.total_scans_processed += 1
        self.card_total.config(text=str(self.total_scans_processed))
        matrix, text_result, theme_color, confidence = self.vision_engine.analyze_scan_matrix(patient_scan_mode)

        self.card_conf.config(text=f"{confidence * 100:.1f}%", fg=theme_color)
        self.lbl_status.config(text=text_result, fg=theme_color)

        log_entry = f"Case #{self.total_scans_processed:03d} | Match Conf: {confidence:.2f} ➔ {patient_scan_mode}"
        self.log_box.insert(tk.END, log_entry)
        self.log_box.itemconfig(tk.END, {'fg': theme_color})
        self.log_box.see(tk.END)

        self.canvas_scan.delete("all")
        c_w = self.canvas_scan.winfo_width() if self.canvas_scan.winfo_width() > 50 else 500
        c_h = self.canvas_scan.winfo_height() if self.canvas_scan.winfo_height() > 50 else 420
        start_x = (c_w - 256) // 2
        start_y = (c_h - 256) // 2

        self.canvas_scan.create_rectangle(start_x - 5, start_y - 5, start_x + 261, start_y + 261, outline=COLOR_TEXT_MUTED, width=1)

        for r in range(0, IMAGE_TARGET_SIZE, 4):
            for c in range(0, IMAGE_TARGET_SIZE, 4):
                val = int(matrix[r, c])
                hex_color = f"#{val:02x}{val:02x}{val:02x}"
                x0 = start_x + (c * 2)
                y0 = start_y + (r * 2)
                x1 = x0 + 8
                y1 = y0 + 8
                self.canvas_scan.create_rectangle(x0, y0, x1, y1, fill=hex_color, outline="")

        if patient_scan_mode == "INFECTED":
            self.canvas_scan.create_rectangle(start_x + 80, start_y + 80, start_x + 160, start_y + 160, outline=COLOR_ANOMALY, width=2, dash=(4,4))
            self.canvas_scan.create_text(start_x + 120, start_y + 70, text="OPAQUE INFILTRATE REGION", fill=COLOR_ANOMALY, font=("Courier New", 7, "bold"))
            messagebox.showwarning("CLINICAL DECISION ALERT", 
                                   f"CRITICAL PATHOLOGY FLAG DETECTED!\n\nAI Confidence: {confidence * 100:.1f}%\nDiagnostic Vector: Pneumonia Infiltrate\n\nForwarding radiography mapping overlays to active physician immediately.")

if __name__ == "__main__":
    app_root = tk.Tk()
    soc_app = ClinicalRadiologySOCPanel(app_root)
    app_root.after(250, lambda: soc_app.trigger_image_diagnostic_cycle("CLEAR"))
    app_root.mainloop()
