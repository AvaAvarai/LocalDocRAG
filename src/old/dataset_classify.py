import tkinter as tk
import pandas as pd
import os
from tkinter import messagebox

class SentenceClassifierApp:
    def __init__(self, master, csv_path):
        self.master = master
        self.master.title("Sentence Classifier")
        
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.index = 0
        self.results = []

        self.label = tk.Label(master, text="Sentence Classifier", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.sentence_label = tk.Label(master, text=self.get_sentence(), wraplength=400, justify="left", font=("Helvetica", 12))
        self.sentence_label.pack(pady=10)

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=20)

        self.useful_button = tk.Button(self.button_frame, text="Useful", command=self.mark_useful, width=15)
        self.useful_button.grid(row=0, column=0, padx=10)

        self.not_useful_button = tk.Button(self.button_frame, text="Not Useful", command=self.mark_not_useful, width=15)
        self.not_useful_button.grid(row=0, column=1, padx=10)

        self.export_button = tk.Button(master, text="Export Results", command=self.export_results, width=20)
        self.export_button.pack(pady=20)

    def get_sentence(self):
        if self.index < len(self.df):
            return self.df.iloc[self.index]['sentence']
        else:
            return "No more sentences."

    def mark_useful(self):
        self.results.append((self.get_sentence(), "useful"))
        self.next_sentence()

    def mark_not_useful(self):
        self.results.append((self.get_sentence(), "not useful"))
        self.next_sentence()

    def next_sentence(self):
        self.index += 1
        if self.index < len(self.df):
            self.sentence_label.config(text=self.get_sentence())
        else:
            self.sentence_label.config(text="No more sentences.")
            self.useful_button.config(state=tk.DISABLED)
            self.not_useful_button.config(state=tk.DISABLED)

    def export_results(self):
        results_df = pd.DataFrame(self.results, columns=['sentence', 'label'])
        export_path = os.path.splitext(self.csv_path)[0] + '_classified.csv'
        results_df.to_csv(export_path, index=False)
        messagebox.showinfo("Export Results", f"Results exported to {export_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentenceClassifierApp(root, 'cleaned_texts.csv')
    root.mainloop()
