import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import sys
import csv
from datetime import datetime

# Определение путей к модели
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "vit_jones19_model")
processor_path = os.path.join(base_path, "vit_jones19_processor")

# Классы культурных стилей (19)
class_names = [
    'Arabian', 'Byzantine', 'Celtic', 'Chinese', 'Egyptian',
    'Elizabethan', 'Greek', 'Hindoo', 'Indian', 'Italian',
    'Medieval', 'Moresque', 'Nineveh & Persia', 'Persian',
    'Pompeian', 'Renaissance', 'Roman', 'Savage Tribes', 'Turkish'
]

# Функция сохранения результатов в CSV (UTF-8 с BOM для Excel)
def save_to_csv(image_path, results):
    """Сохраняет результат классификации в CSV файл (UTF-8-sig для Excel)"""
    csv_file = "classification_history.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow(['Дата и время', 'Файл', 'Топ-1', 'Уверенность', 'Топ-2', 'Топ-3'])
        
        top1_name, top1_prob = results[0]
        top2_name, top2_prob = results[1]
        top3_name, top3_prob = results[2]
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(image_path),
            top1_name, f"{top1_prob:.1%}",
            f"{top2_name} ({top2_prob:.1%})",
            f"{top3_name} ({top3_prob:.1%})"
        ])

# Загрузка модели
print("Загрузка модели...")
processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
model.eval()
print("Модель загружена")

# Функция предсказания
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top3_probs, top3_indices = torch.topk(probs, 3)
    top3_probs = top3_probs.numpy()[0]
    top3_indices = top3_indices.numpy()[0]
    results = [(class_names[idx], float(prob)) for idx, prob in zip(top3_indices, top3_probs)]
    return results

# GUI приложения
class App:
    def __init__(self, root):
        self.root = root
        root.title("Классификатор культурных орнаментов")
        root.geometry("750x650")
        root.configure(bg='#2c3e50')
        root.resizable(True, True)
        
        # Цветовая схема
        self.colors = {
            'bg': '#2c3e50',
            'fg': '#ecf0f1',
            'accent': '#e67e22',
            'button': '#3498db',
            'button_hover': '#2980b9',
            'result_bg': '#34495e',
            'export_btn': '#27ae60',
            'export_hover': '#219a52'
        }
        
        # Заголовок
        title = tk.Label(root, text="Классификатор культурных орнаментов", 
                        font=('Segoe UI', 16, 'bold'), 
                        bg=self.colors['bg'], fg=self.colors['fg'])
        title.pack(pady=15)
        
        # Подзаголовок
        subtitle = tk.Label(root, text="Загрузите изображение орнамента - нейросеть определит стиль",
                           font=('Segoe UI', 10), 
                           bg=self.colors['bg'], fg=self.colors['fg'])
        subtitle.pack(pady=5)
        
        # Панель кнопок
        self.btn_frame = tk.Frame(root, bg=self.colors['bg'])
        self.btn_frame.pack(pady=15)
        
        # Кнопка загрузки
        self.btn_load = tk.Button(self.btn_frame, text="Выбрать изображение", 
                                 command=self.load_image,
                                 font=('Segoe UI', 11, 'bold'),
                                 bg=self.colors['button'], fg='white',
                                 padx=20, pady=8,
                                 relief=tk.FLAT, cursor='hand2',
                                 activebackground=self.colors['button_hover'],
                                 activeforeground='white')
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # Эффект наведения
        self.btn_load.bind("<Enter>", lambda e: self.btn_load.config(bg=self.colors['button_hover']))
        self.btn_load.bind("<Leave>", lambda e: self.btn_load.config(bg=self.colors['button']))
        
        # Кнопка экспорта CSV
        self.btn_export = tk.Button(self.btn_frame, text="Экспорт CSV", 
                                   command=self.export_csv,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.colors['export_btn'], fg='white',
                                   padx=20, pady=8,
                                   relief=tk.FLAT, cursor='hand2',
                                   activebackground=self.colors['export_hover'],
                                   activeforeground='white')
        self.btn_export.pack(side=tk.LEFT, padx=5)
        
        self.btn_export.bind("<Enter>", lambda e: self.btn_export.config(bg=self.colors['export_hover']))
        self.btn_export.bind("<Leave>", lambda e: self.btn_export.config(bg=self.colors['export_btn']))
        
        # Область для изображения
        self.image_frame = tk.Frame(root, bg=self.colors['bg'], relief=tk.GROOVE, bd=2)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(self.image_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.image_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Рамка для результатов
        self.result_frame = tk.LabelFrame(root, text="Результат классификации", 
                                         font=('Segoe UI', 10, 'bold'),
                                         bg=self.colors['bg'], fg=self.colors['fg'],
                                         relief=tk.GROOVE, bd=2)
        self.result_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.result_text = tk.Text(self.result_frame, height=8, width=60,
                                  font=('Consolas', 11),
                                  bg=self.colors['result_bg'], fg=self.colors['fg'],
                                  relief=tk.FLAT, wrap=tk.WORD)
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Статус бар
        self.status = tk.Label(root, text="Готов к работе", 
                              font=('Segoe UI', 9), 
                              bg=self.colors['bg'], fg=self.colors['fg'],
                              anchor='w')
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)
    
    def export_csv(self):
        """Открывает папку с CSV файлом"""
        csv_file = "classification_history.csv"
        if os.path.exists(csv_file):
            os.startfile(csv_file)
        else:
            messagebox.showinfo("Информация", "История пока пуста. Сначала выполните классификацию.")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
            title="Выберите изображение орнамента"
        )
        if file_path:
            self.status.config(text="Обработка изображения...")
            self.root.update()
            
            # Показываем изображение
            img = Image.open(file_path)
            img.thumbnail((500, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Предсказание
            results = predict_image(file_path)
            
            # Сохраняем в CSV
            save_to_csv(file_path, results)
            
            # Форматируем вывод
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "\n")
            for i, (name, prob) in enumerate(results, 1):
                bar_len = int(prob * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                self.result_text.insert(tk.END, f"  {i}. {name:<25} {bar} {prob:.1%}\n\n")
            
            self.status.config(text="Готов")

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()