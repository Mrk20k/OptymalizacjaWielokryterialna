import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import metody_optymalizacji
import tkinter.font as tkFont

def fmt_sig(x, sig=4):
    if isinstance(x, (int, float)):
        if x == 0:
            return "0"
        return f"{x:.{sig}g}"
    return str(x)

def fmt_tuple(t, sig=4):
    return tuple(float(fmt_sig(x, sig)) for x in t)
# ============================
#       MAIN GUI CLASS
# ============================
class ParetoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pareto Explorer 3D")
        self.geometry("1200x700")

        self.num_criteria = 3
        self.kierunki = [1]*self.num_criteria
        self.data = []
        self.create_widgets()
        self.generate_random_data()
        self.update_table()

    # --- GUI ---
    def create_widgets(self):
        left = ttk.Frame(self)
        left.pack(side='left', fill='y', padx=8, pady=8)

        ttk.Label(left, text='Edytor kryteriów').pack()
        self.criteria_frame = ttk.Frame(left)
        self.criteria_frame.pack(pady=4, fill='x')
        self.criteria_rows = []
        self.build_criteria_editor()

        # generacja danych
        gen_frame = ttk.LabelFrame(left, text='Generacja danych')
        gen_frame.pack(fill='x', pady=6)
        ttk.Label(gen_frame, text='Rozkład:').grid(row=0, column=0)
        self.dist_cb = ttk.Combobox(gen_frame, values=['Normalny', 'Jednostajny', 'Eksponencjalny'], state='readonly')
        self.dist_cb.current(1)
        self.dist_cb.grid(row=0, column=1)

        ttk.Label(gen_frame, text='Parametr:').grid(row=1, column=0)
        self.param_entry = ttk.Entry(gen_frame)
        self.param_entry.insert(0, '5')
        self.param_entry.grid(row=1, column=1)

        ttk.Label(gen_frame, text='Liczba obiektów:').grid(row=2, column=0)
        self.count_entry = ttk.Entry(gen_frame)
        self.count_entry.insert(0, '10')
        self.count_entry.grid(row=2, column=1)

        ttk.Label(gen_frame, text='Liczba zbiorów:').grid(row=3, column=0)
        self.num_sets_entry = ttk.Entry(gen_frame)
        self.num_sets_entry.insert(0, '1')
        self.num_sets_entry.grid(row=3, column=1)

        ttk.Button(gen_frame, text='Generuj', command=self.on_generate).grid(row=4, column=0, columnspan=2, pady=4)

        # wybór zbioru (dla wielu zbiorów)
        ttk.Label(gen_frame, text='Podgląd zbioru:').grid(row=5, column=0)
        self.dataset_select = ttk.Combobox(gen_frame, state='disabled', width=5)
        self.dataset_select.grid(row=5, column=1)
        self.dataset_select.bind('<<ComboboxSelected>>', self.on_dataset_change)

        # import danych
        imp_frame = ttk.LabelFrame(left, text='Import danych')
        imp_frame.pack(fill='x', pady=6)
        ttk.Button(imp_frame, text='Wczytaj CSV', command=self.load_csv).pack(fill='x', pady=2)
        ttk.Button(imp_frame, text='Wczytaj Excel', command=self.load_excel).pack(fill='x', pady=2)

        # tabela
        center = ttk.Frame(self)
        center.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        ttk.Label(center, text='Dane').pack()

        table_container = ttk.Frame(center)
        table_container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(table_container)
        self.canvas.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(table_container, orient='vertical', command=self.canvas.yview)
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(center, orient='horizontal', command=self.canvas.xview)
        hsb.pack(side='bottom', fill='x')

        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.table = ttk.Treeview(self.scrollable_frame, show='headings', height=20)
        self.table.pack(fill='both', expand=True)
        self.make_table_editable()

        tb_buttons = ttk.Frame(center)
        tb_buttons.pack(fill='x')
        ttk.Button(tb_buttons, text='Dodaj wiersz', command=self.add_row).pack(side='left')
        ttk.Button(tb_buttons, text='Usuń wiersz', command=self.del_row).pack(side='left')

        # prawa strona
        right = ttk.Frame(self)
        right.pack(side='right', fill='y', padx=8, pady=8)

        alg_frame = ttk.LabelFrame(right, text='Algorytm')
        alg_frame.pack(fill='x')
        self.alg_cb = ttk.Combobox(alg_frame, values=['Naiwny','Filtracja','Punkt idealny'], state='readonly')
        self.alg_cb.current(0)
        self.alg_cb.grid(row=0, column=0)
        ttk.Button(alg_frame, text='Rozwiąż', command=self.on_solve).grid(row=1, column=0, pady=5)

        self.results_text = tk.Text(right, width=45, height=20)
        self.results_text.pack(pady=10)

    # --- KRYTERIA ---
    def build_criteria_editor(self):
        for w in self.criteria_frame.winfo_children():
            w.destroy()
        self.criteria_rows.clear()
        for i in range(self.num_criteria):
            ttk.Label(self.criteria_frame, text=f'Kryterium {i+1}').grid(row=i, column=0)
            cb = ttk.Combobox(self.criteria_frame, values=['Min','Max'], width=5, state='readonly')
            cb.current(0 if self.kierunki[i]==1 else 1)
            cb.grid(row=i, column=1)
            cb.bind('<<ComboboxSelected>>', partial(self.on_direction_change, i))
            self.criteria_rows.append(cb)

        btns = ttk.Frame(self.criteria_frame)
        btns.grid(row=self.num_criteria, column=0, columnspan=2, pady=4)
        ttk.Button(btns, text='Dodaj', command=self.add_criterion).pack(side='left')
        ttk.Button(btns, text='Usuń', command=self.remove_criterion).pack(side='left')

    def on_direction_change(self, idx, event=None):
        val = self.criteria_rows[idx].get()
        self.kierunki[idx] = 1 if val == 'Min' else -1

    def add_criterion(self):
        self.num_criteria += 1
        self.kierunki.append(1)
        
        # generate new values for the new column
        n = len(self.data)
        param = float(self.param_entry.get() or 5)
        dist = self.dist_cb.get()
        
        new_vals = []
        if dist == 'Normalny':
            new_vals = list(np.random.normal(loc=param, scale=max(1, param/0.3), size=n))
        elif dist == 'Jednostajny':
            new_vals = list(np.random.uniform(0, param, size=n))
        else:
            new_vals = list(np.random.exponential(scale=max(1,param), size=n))

        self.data = [tuple(list(row) + [v]) for row,v in zip(self.data, new_vals)]

        self.build_criteria_editor()
        self.update_table()
        self.make_table_editable()

    def remove_criterion(self):
        if self.num_criteria > 1:
            self.num_criteria -= 1
            self.kierunki.pop()
            self.build_criteria_editor()
            self.update_table()

    # --- DANE ---
    def generate_random_data(self):
        n = 10
        m = 5
        self.data = [tuple(np.random.normal(loc=m, scale=3.0, size=self.num_criteria)) for _ in range(n)]

    def on_generate(self):
        try:
            n = int(self.count_entry.get())
            param = float(self.param_entry.get())
            num_sets = int(self.num_sets_entry.get())
        except ValueError:
            messagebox.showerror('Błąd', 'Niepoprawne parametry generacji')
            return

        dist = self.dist_cb.get()

        def generate_one():
            if dist == 'Normalny':
                return [tuple(np.random.normal(loc=param, scale=max(1, param), size=self.num_criteria)) for _ in range(n)]
            elif dist == 'Jednostajny':
                return [tuple(np.random.uniform(0, param, size=self.num_criteria)) for _ in range(n)]
            else:
                return [tuple(np.random.exponential(scale=max(1,param), size=self.num_criteria)) for _ in range(n)]

        if num_sets > 1:
            self.data = [generate_one() for _ in range(num_sets)]
            self.current_dataset = 0

            # włącz combobox wyboru zbioru
            self.dataset_select['values'] = [str(i+1) for i in range(num_sets)]
            self.dataset_select.current(0)
            self.dataset_select['state'] = 'readonly'

            self.update_table_with_dataset(self.data[self.current_dataset])
            messagebox.showinfo(
                'Info',
                f'Wygenerowano {num_sets} zbiorów danych, każdy po {n} punktów.\n'
                f'Możesz przeglądać je z listy "Podgląd zbioru".\n'
                f'Dla wielu zbiorów punkty Pareto i wykres nie będą wyświetlane.'
            )
        else:
            self.data = generate_one()
            self.dataset_select.set('')
            self.dataset_select['state'] = 'disabled'
            self.update_table()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
            self.data = [tuple(x) for x in df.values.tolist()]
            self.num_criteria = len(df.columns)
            self.kierunki = [1]*self.num_criteria
            self.build_criteria_editor()
            self.update_table()
        except Exception as e:
            messagebox.showerror('Błąd', f'Nie udało się wczytać pliku:\n{e}')

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if not path: return
        try:
            df = pd.read_excel(path)
            self.data = [tuple(x) for x in df.values.tolist()]
            self.num_criteria = len(df.columns)
            self.kierunki = [1]*self.num_criteria
            self.build_criteria_editor()
            self.update_table()
        except Exception as e:
            messagebox.showerror('Błąd', f'Nie udało się wczytać pliku:\n{e}')

    def render_table(self, dataset):
        """Aktualizuje zawartość tabeli dla dowolnego zbioru danych."""
        self.table["columns"] = [f"c{i}" for i in range(1, self.num_criteria + 1)]

        # oblicz szerokość kolumn dynamicznie, z limitem
        max_width_per_col = 120
        font = tkFont.Font()

        # upewnij się, że dataset nie jest pusty
        if not dataset:
            dataset = []

        for i, col in enumerate(self.table["columns"]):
            header_text = f"Kryterium {i+1}"
            max_text_width = max(
                [font.measure(str(row[i])) for row in dataset] + [font.measure(header_text)] if dataset else [font.measure(header_text)]
            )
            width = min(max_text_width + 20, max_width_per_col)
            self.table.heading(col, text=header_text)
            self.table.column(col, width=width, minwidth=50, anchor='center', stretch=False)

        # usuń poprzednie wiersze
        for r in self.table.get_children():
            self.table.delete(r)

        # wstaw nowe dane
        for idx, row in enumerate(dataset):
            padded = list(row) + [''] * (self.num_criteria - len(row))
            formatted = [fmt_sig(v) if isinstance(v, (int, float)) else v for v in padded]
            self.table.insert('', 'end', iid=str(idx), values=formatted)

        # odśwież scrollbar
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def update_table(self):
        """Aktualizuje tabelę dla pojedynczego zbioru danych."""
        if isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], (list, tuple)):
            self.render_table(self.data)
        else:
            self.render_table([])


    def update_table_with_dataset(self, dataset):
        """Aktualizuje tabelę przy wielu zbiorach danych (zachowując scroll)."""
        self.render_table(dataset)

    def on_dataset_change(self, event=None):
        if not hasattr(self, 'data') or not self.data:
            return
        try:
            idx = int(self.dataset_select.get()) - 1
            if 0 <= idx < len(self.data):
                self.current_dataset = idx
                self.update_table_with_dataset(self.data[idx])
        except Exception:
            pass

    def add_row(self):
        vals = [0.0]*self.num_criteria
        self.data.append(tuple(vals))
        self.update_table()

    def del_row(self):
        sel = self.table.selection()
        if not sel:
            messagebox.showinfo('Info', 'Wybierz wiersz do usunięcia')
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.data):
            self.data.pop(idx)
        self.update_table()

    # --- ALGORYTMY ---
    def on_solve(self):
        if not self.data:
            messagebox.showinfo('Info', 'Brak danych')
            return

        alg = self.alg_cb.get()

        # Sprawdź, czy mamy wiele zbiorów
        if isinstance(self.data[0][0], (list, tuple)):
            total_p = 0
            total_points = 0
            total_por_p = 0
            total_por_wsp = 0
            total_time = 0.0

            for dataset in self.data:
                X = [tuple(map(float, p)) for p in dataset]
                if alg == 'Naiwny':
                    P, por_p, por_wsp, czas = metody_optymalizacji.znajdz_front_pareto(X, self.kierunki)
                elif alg == 'Filtracja':
                    P, por_p, por_wsp, czas = metody_optymalizacji.znajdz_front_z_filtracja(X, self.kierunki)
                else:
                    P, por_p, por_wsp, czas, _ = metody_optymalizacji.algorytm_punkt_idealny(X, self.kierunki)

                total_points += len(X)
                total_p += len(P)
                total_por_p += por_p
                total_por_wsp += por_wsp
                total_time += czas

            self.results_text.delete('1.0', tk.END)
            self.results_text.insert(tk.END, f'=== Wiele zbiorów danych ===\n')
            self.results_text.insert(tk.END, f'Liczba zbiorów: {len(self.data)}\n')
            self.results_text.insert(tk.END, f'Łączna liczba punktów: {total_points}\n')
            self.results_text.insert(tk.END, f'Łączna liczba punktów Pareto: {total_p}\n')
            self.results_text.insert(tk.END, f'Łączna liczba porównań punktów: {total_por_p}\n')
            self.results_text.insert(tk.END, f'Łączna liczba porównań współrzędnych: {total_por_wsp}\n')
            self.results_text.insert(tk.END, f'Łączny czas obliczeń: {total_time:.6f} s\n\n')
            self.results_text.insert(tk.END, 'Dla wielu zbiorów danych punkty Pareto nie są wyświetlane.\n')
            return

        # --- tryb pojedynczego zbioru ---
        X = [tuple(map(float, p)) for p in self.data]
        punkt_idealny = []

        if alg == 'Naiwny':
            P, por_p, por_wsp, czas = metody_optymalizacji.znajdz_front_pareto(X, self.kierunki)
        elif alg == 'Filtracja':
            P, por_p, por_wsp, czas = metody_optymalizacji.znajdz_front_z_filtracja(X, self.kierunki)
        else:
            P, por_p, por_wsp, czas, punkt_idealny = metody_optymalizacji.algorytm_punkt_idealny(X, self.kierunki)

        remaining = len(X) - len(P)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, f'Front Pareto: {len(P)} punktów\n')
        self.results_text.insert(tk.END, f'Zdominowanych: {remaining}\n')
        self.results_text.insert(tk.END, f'Porównań punktów: {por_p}\n')
        self.results_text.insert(tk.END, f'Porównań współrzędnych: {por_wsp}\n')
        self.results_text.insert(tk.END, f'Czas: {czas:.6f} s\n\n')

        if punkt_idealny:
            self.results_text.insert(tk.END, 'Punkt idealny:\n')
            self.results_text.insert(tk.END, f'{fmt_tuple(punkt_idealny)}\n')
        self.results_text.insert(tk.END, 'Punkty Pareto:\n')
        for p in P:
            self.results_text.insert(tk.END, f'{fmt_tuple(p)}\n')

        # wykres tylko dla pojedynczego zbioru
        if punkt_idealny:
            self.show_plot_window(X, P, punkt_idealny)
        else:
            self.show_plot_window(X, P)


    # --- NOWE OKNO Z WYKRESEM 3D ---
    def show_plot_window(self, X, P, punkt_idealny=[]):
        win = tk.Toplevel(self)
        win.title("Wykres frontu Pareto")
        fig = Figure(figsize=(6,5))

        if len(X[0]) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5,0.5,'Brak dwóch wymiarów do wykresu', ha='center')
        elif len(X[0]) == 2:
            ax = fig.add_subplot(111)
            # filter X to exclude Pareto points
            X_non_P = [x for x in X if x not in P]
            X_arr = np.array(X_non_P)
            if len(X_arr) > 0:
                ax.scatter(X_arr[:,0], X_arr[:,1], alpha=0.6, label='Punkty zdominowane')
            if P:
                P_arr = np.array(P)
                ax.scatter(P_arr[:,0], P_arr[:,1], color='red', label='Front Pareto')
            if punkt_idealny:
                ax.scatter(punkt_idealny[0], punkt_idealny[1], color='purple', label='Punkt idealny')
            ax.set_xlabel('Kryterium 1')
            ax.set_ylabel('Kryterium 2')
            ax.legend()
        else:
            ax = fig.add_subplot(111, projection='3d')
            # filter X to exclude Pareto points
            X_non_P = [x for x in X if x not in P]
            X_arr = np.array(X_non_P)
            if len(X_arr) > 0:
                ax.scatter(X_arr[:,0], X_arr[:,1], X_arr[:,2], alpha=0.6, label='Punkty zdominowane')
            if P:
                P_arr = np.array(P)
                ax.scatter(P_arr[:,0], P_arr[:,1], P_arr[:,2], color='red', label='Front Pareto')
            if punkt_idealny:
                ax.scatter(punkt_idealny[0], punkt_idealny[1], punkt_idealny[2], color='purple', label='Punkt idealny')
            ax.set_xlabel('Kryterium 1')
            ax.set_ylabel('Kryterium 2')
            ax.set_zlabel('Kryterium 3')
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

    def make_table_editable(self):
        self.table.bind('<Double-1>', self.on_cell_double_click)

    def on_cell_double_click(self, event):
        region = self.table.identify('region', event.x, event.y)
        if region != 'cell':
            return

        row_id = self.table.identify_row(event.y)
        col = self.table.identify_column(event.x)
        col_idx = int(col[1:]) - 1  # columns are '#1', '#2', etc.

        # current value
        cur_val = self.table.set(row_id, self.table["columns"][col_idx])

        # place an Entry widget
        x, y, width, height = self.table.bbox(row_id, col)
        entry = tk.Entry(self.table)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, cur_val)
        entry.focus()

        def save_edit(event=None):
            val = entry.get()
            entry.destroy()
            self.table.set(row_id, self.table["columns"][col_idx], val)
            # update internal data
            self.data[int(row_id)] = tuple(
                float(val) if i==col_idx else v for i,v in enumerate(self.data[int(row_id)])
            )

        entry.bind('<Return>', save_edit)
        entry.bind('<FocusOut>', save_edit)

if __name__ == '__main__':
    app = ParetoApp()
    app.mainloop()
