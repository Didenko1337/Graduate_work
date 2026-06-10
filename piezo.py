"""
Моделирование гармонических колебаний пьезоэлектрического элемента (юниморф)
"""

import sys
import numpy as np
from scipy.special import j0, j1, i0e, i1e
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QDoubleSpinBox, QSpinBox, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

MATERIALS = {
    "PZT-4": {
        "c11E": 13.9e10,    # Па
        "c12E": 7.78e10,
        "c13E": 7.43e10,
        "c33E": 11.5e10,
        "c44E": 2.56e10,
        "e31":  -5.2,       # Кл/м²
        "e33":  15.1,
        "e15":  12.7,
        "eps33": 562e-12,   # Ф/м
        "eps11": 645e-12,
        "rho":  7500.0,     # кг/м³
    },
    "PZT-5A": {
        "c11E": 12.1e10,
        "c12E": 7.54e10,
        "c13E": 7.52e10,
        "c33E": 11.1e10,
        "c44E": 2.11e10,
        "e31":  -5.35,
        "e33":  15.78,
        "e15":  12.29,
        "eps33": 830e-12,
        "eps11": 916e-12,
        "rho":  7750.0,
    },
    "BaTiO3": {
        "c11E": 15.0e10,
        "c12E": 6.60e10,
        "c13E": 6.60e10,
        "c33E": 14.6e10,
        "c44E": 4.30e10,
        "e31":  -4.35,
        "e33":  17.50,
        "e15":  11.40,
        "eps33": 1260e-12,
        "eps11": 1115e-12,
        "rho":  5700.0,
    },
}


class PiezoModel:
    """Аналитическая модель юниморфа"""

    def __init__(self, mat: dict, radius_a: float, thickness_h: float, voltage_v0: float):
        self.mat = mat
        self.a = radius_a
        self.h = thickness_h
        self.v0 = voltage_v0

        self.c11s = 0.0
        self.c12s = 0.0
        self.e31s = 0.0
        self.e15s = 0.0
        self.nu2 = 0.0
        self.nu = 0.0
        self.mu = 0.0

        self._compute_effective()

    def _compute_effective(self):
        m = self.mat
        c11e = m["c11E"]
        c12e = m["c12E"]
        c13e = m["c13E"]
        c33e = m["c33E"]
        c44e = m["c44E"]
        e31  = m["e31"]
        e33  = m["e33"]
        e15  = m["e15"]
        eps33 = m["eps33"]
        eps11 = m["eps11"]

        # модифицированные упругие константы (плоское напряжённое состояние)
        self.c11s = c11e * (1.0 - (c13e**2) / (c11e * c33e))
        self.c12s = c12e * (1.0 - (c13e**2) / (c12e * c33e))
        # эффективные пьезоэлектрические модули
        self.e31s = e31 * (1.0 - c13e * e33 / (c33e * e31))
        self.e15s = e15 * (1.0 - c44e * e33 / (c33e * e15))
        # параметры для уравнения потенциала
        self.nu2  = eps11 / eps33         # nu^2 = eps11/eps33
        self.nu   = np.sqrt(abs(self.nu2))
        self.mu   = self.e15s / eps33

    def kappa(self, omega: float) -> float:
        """Волновое число κ = sqrt(ρ ω² / c11*)"""
        return np.sqrt(self.mat["rho"] * omega**2 / self.c11s)

    def delta_det(self, kappa: float) -> float:
        """Характеристический определитель Δ(κa)"""
        ka = kappa * self.a
        return self.c11s * ka * j0(ka) - (self.c11s - self.c12s) * j1(ka)

    def radial_displacement(self, r: np.ndarray, omega: float) -> np.ndarray:
        """
        u(r) = (2 V0 a e31*) / (h Δ(κa)) * J1(κr)
        """
        kap = self.kappa(omega)
        denom = self.delta_det(kap)
        if abs(denom) < 1e-30:
            return np.zeros_like(r)
        amp = 2.0 * self.v0 * self.a * self.e31s / (self.h * denom)
        return amp * j1(kap * r)

    def electric_potential(self, r: np.ndarray, coordinate_z: float, omega: float,
                           n_terms: int = 40) -> np.ndarray:
        """
        φ(r,z) = φ1(r,z) + φ2(z)
        φ2 = 2 V0 z / h
        φ1 — ряд по собственным функциям sin(2πn z / h)
        """
        h = self.h
        v0 = self.v0
        kap = self.kappa(omega)
        denom = self.delta_det(kap)
        if abs(denom) < 1e-30:
            return 2.0 * v0 * coordinate_z / h * np.ones_like(r)

        amp = 2.0 * v0 * self.a * self.e31s / (h * denom)
        phi2 = 2.0 * v0 * coordinate_z / h

        phi1 = np.zeros_like(r, dtype=float)
        for n in range(1, n_terms + 1):
            lam_n = 2.0 * np.pi * n / h
            sin_z = np.sin(lam_n * coordinate_z)
            if abs(sin_z) < 1e-50:
                continue

            # частное решение — коэффициент при J0
            denom_n = kap**2 + self.nu2 * lam_n**2
            if abs(denom_n) < 1e-50:
                continue
            d1 = self.mu * amp * (h / (np.pi * n)) * ((-1)**n) * kap**3 / denom_n
            a_arg = self.nu * self.a * lam_n
            i1e_a = i1e(a_arg)
            if abs(i1e_a) < 1e-300:
                dn = 0.0
            else:
                factor = self.mu * amp * (h**2 / (np.pi**2 * 2.0 * self.nu))
                dn = (factor * ((-1) ** (n + 1)) * j1(kap * self.a)
                      * (kap ** 2 / (n ** 2 + (kap * h / (self.nu * 2.0 * np.pi * n)) ** 2)))
            r_arg = self.nu * r * lam_n
            ratio_i = i0e(r_arg) / i1e_a * np.exp((r - self.a) * self.nu * lam_n)

            g_n = d1 * j0(kap * r) + dn * ratio_i
            phi1 += g_n * sin_z

        return phi2 + phi1

    def find_resonances(self, f_min=1e3, f_max=1e6, n_pts=5000) -> list:
        """Поиск резонансных частот (нули Δ(κa))"""
        freqs = np.linspace(f_min, f_max, n_pts)
        omegas = 2.0 * np.pi * freqs
        resonances = []
        deltas = [self.delta_det(self.kappa(w)) for w in omegas]
        for i in range(len(deltas) - 1):
            if deltas[i] * deltas[i + 1] < 0:
                try:
                    w_res = brentq(lambda w: self.delta_det(self.kappa(w)),
                                   omegas[i], omegas[i + 1], xtol=1e-3)
                    resonances.append(w_res / (2.0 * np.pi))
                except LookupError:
                    pass
        return resonances

    def frequency_response(self, f_min=1e3, f_max=1e6, n_pts=500) -> tuple:
        """АЧХ: максимальное смещение vs частота"""
        freqs = np.linspace(f_min, f_max, n_pts)
        r_mid = self.a * 0.5
        u_max = []
        for f in freqs:
            w = 2.0 * np.pi * f
            u_mid = abs(self.radial_displacement(np.array([r_mid]), w)[0])
            u_max.append(u_mid)
        return freqs, np.array(u_max)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(9, 5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.axes = []
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            self.axes.append(ax)
        self.setMinimumHeight(300)


class ParamBox(QGroupBox):
    """Компактный блок ввода одного параметра"""
    def __init__(self, label: str, value: float, unit: str,
                 lo: float, hi: float, decimals: int = 4, parent=None):
        super().__init__(parent)
        self.setTitle("")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lbl = QLabel(f"<b>{label}</b>")
        lbl.setMinimumWidth(70)
        lay.addWidget(lbl)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(value)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep((hi - lo) / 100)
        self.spin.setMinimumWidth(100)
        lay.addWidget(self.spin)
        lay.addWidget(QLabel(unit))

    @property
    def value(self) -> float:
        return self.spin.value()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Явное объявление атрибутов экземпляра класса во избежание linter-warnings
        self._model = None
        self._r = np.array([])
        self._u = np.array([])
        self._omega = 0.0
        self._freqs = np.array([])
        self._u_freq = np.array([])
        self._res_list = []
        self._sigma_rr = np.array([])

        self.cb_material = None
        self.w_a = None
        self.w_h = None
        self.w_v0 = None
        self.w_freq = None
        self.w_fmin = None
        self.w_fmax = None
        self.w_npts = None
        self.spin_nterms = None
        self.btn_compute = None
        self.btn_save = None
        self.info_box = None
        self.lbl_kappa = None
        self.lbl_delta = None
        self.lbl_u_max = None
        self.lbl_res = None
        self.tabs = None
        self.canvas1 = None
        self.canvas2 = None
        self.w_z_norm = None
        self.canvas3 = None
        self.canvas4 = None
        self.canvas5 = None
        self.canvas6 = None

        self.setWindowTitle("Моделирование гармонических колебаний пьезоэлектрического элемента (юниморф)")
        self.resize(1350, 820)
        self._build_ui()
        self._apply_style()
        self.on_compute()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)

        # параметры
        left = QWidget()
        left.setMaximumWidth(310)
        left.setMinimumWidth(260)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(6)

        # материал
        g_mat = QGroupBox("Материал")
        g_mat_lay = QVBoxLayout(g_mat)
        self.cb_material = QComboBox()
        self.cb_material.addItems(list(MATERIALS.keys()))
        self.cb_material.currentTextChanged.connect(self._on_material_change)
        g_mat_lay.addWidget(self.cb_material)
        left_lay.addWidget(g_mat)

        # геометрия
        g_geo = QGroupBox("Геометрия пластины")
        g_geo_lay = QVBoxLayout(g_geo)
        self.w_a  = ParamBox("a (радиус)", 0.020,  "м",  1e-4, 0.5, decimals=4)
        self.w_h  = ParamBox("h (толщина)", 0.001, "м",  1e-5, 0.05, decimals=5)
        g_geo_lay.addWidget(self.w_a)
        g_geo_lay.addWidget(self.w_h)
        left_lay.addWidget(g_geo)

        # возбуждение
        g_exc = QGroupBox("Возбуждение")
        g_exc_lay = QVBoxLayout(g_exc)
        self.w_v0   = ParamBox("V₀",         1.0,    "В",    0.001, 1000, decimals=3)
        self.w_freq = ParamBox("f",           50000,  "Гц",   100,   1e6,  decimals=0)
        g_exc_lay.addWidget(self.w_v0)
        g_exc_lay.addWidget(self.w_freq)
        left_lay.addWidget(g_exc)

        # диапазон
        g_frange = QGroupBox("Диапазон частот (АЧХ)")
        g_frange_lay = QVBoxLayout(g_frange)
        self.w_fmin = ParamBox("f_min",  1000,  "Гц",  100,  2e5, decimals=0)
        self.w_fmax = ParamBox("f_max", 500000, "Гц",  1000, 2e6, decimals=0)
        self.w_npts = QGroupBox("")
        npts_lay = QHBoxLayout(self.w_npts)
        npts_lay.setContentsMargins(4, 2, 4, 2)
        npts_lay.addWidget(QLabel("<b>N_terms</b>"))
        self.spin_nterms = QSpinBox()
        self.spin_nterms.setRange(5, 100)
        self.spin_nterms.setValue(25)
        npts_lay.addWidget(self.spin_nterms)
        npts_lay.addWidget(QLabel("членов ряда"))
        g_frange_lay.addWidget(self.w_fmin)
        g_frange_lay.addWidget(self.w_fmax)
        g_frange_lay.addWidget(self.w_npts)
        left_lay.addWidget(g_frange)

        # кнопки
        btn_row = QHBoxLayout()
        self.btn_compute = QPushButton("▶  Рассчитать")
        self.btn_compute.setMinimumHeight(36)
        self.btn_compute.clicked.connect(self.on_compute)
        btn_row.addWidget(self.btn_compute)
        self.btn_save = QPushButton("💾  Сохранить")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.clicked.connect(self.on_save)
        btn_row.addWidget(self.btn_save)
        left_lay.addLayout(btn_row)

        # информационная панель
        self.info_box = QGroupBox("Результаты")
        info_lay = QVBoxLayout(self.info_box)
        self.lbl_kappa = QLabel("κ: —")
        self.lbl_delta = QLabel("Δ(κa): —")
        self.lbl_u_max = QLabel("u_max: —")
        self.lbl_res   = QLabel("Резонансы: —")
        self.lbl_res.setWordWrap(True)
        for lb in [self.lbl_kappa, self.lbl_delta, self.lbl_u_max, self.lbl_res]:
            lb.setFont(QFont("Courier", 9))
            info_lay.addWidget(lb)
        left_lay.addWidget(self.info_box)
        left_lay.addStretch()

        root.addWidget(left)

        self.tabs = QTabWidget()

        # первая вкладка
        tab1 = QWidget()
        t1l = QVBoxLayout(tab1)
        self.canvas1 = MplCanvas(figsize=(9, 4))
        tb1 = NavigationToolbar(self.canvas1, tab1)
        t1l.addWidget(tb1)
        t1l.addWidget(self.canvas1)
        self.tabs.addTab(tab1, "Радиальное смещение u(r)")

        # вторая вкладка
        tab2 = QWidget()
        t2l = QVBoxLayout(tab2)
        self.canvas2 = MplCanvas(figsize=(9, 4))
        tb2 = NavigationToolbar(self.canvas2, tab2)
        t2l.addWidget(tb2)
        t2l.addWidget(self.canvas2)
        self.tabs.addTab(tab2, "АЧХ")

        # третья вкладка
        tab3 = QWidget()
        t3l = QVBoxLayout(tab3)
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("z / (h/2):"))
        self.w_z_norm = QDoubleSpinBox()
        self.w_z_norm.setRange(-1.0, 1.0)
        self.w_z_norm.setValue(0.5)
        self.w_z_norm.setSingleStep(0.1)
        self.w_z_norm.setDecimals(2)
        self.w_z_norm.valueChanged.connect(self._replot_phi)
        z_row.addWidget(self.w_z_norm)
        z_row.addStretch()
        t3l.addLayout(z_row)
        self.canvas3 = MplCanvas(figsize=(9, 4))
        tb3 = NavigationToolbar(self.canvas3, tab3)
        t3l.addWidget(tb3)
        t3l.addWidget(self.canvas3)
        self.tabs.addTab(tab3, "Потенциал φ(r, z)")

        # четвертая вкладка
        tab4 = QWidget()
        t4l = QVBoxLayout(tab4)
        self.canvas4 = MplCanvas(figsize=(9, 5))
        tb4 = NavigationToolbar(self.canvas4, tab4)
        t4l.addWidget(tb4)
        t4l.addWidget(self.canvas4)
        self.tabs.addTab(tab4, "Карта φ(r, z)")

        # пятая вкладка
        tab5 = QWidget()
        t5l = QVBoxLayout(tab5)
        self.canvas5 = MplCanvas(figsize=(9, 4))
        tb5 = NavigationToolbar(self.canvas5, tab5)
        t5l.addWidget(tb5)
        t5l.addWidget(self.canvas5)
        self.tabs.addTab(tab5, "Напряжение σ_rr(r)")

        # шестая вкладка
        tab6 = QWidget()
        t6l = QVBoxLayout(tab6)
        self.canvas6 = MplCanvas(figsize=(9, 4))
        tb6 = NavigationToolbar(self.canvas6, tab6)
        t6l.addWidget(tb6)
        t6l.addWidget(self.canvas6)
        self.tabs.addTab(tab6, "Форма колебаний")

        root.addWidget(self.tabs)

        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage("Готово")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QWidget { background: #1e1e2e; color: #cdd6f4; font-size: 13px; }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 6px;
                font-weight: bold; color: #89b4fa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background: #313244; border: 1px solid #45475a;
                border-radius: 4px; padding: 2px 6px;
                color: #cdd6f4;
            }
            QPushButton {
                background: #89b4fa; color: #1e1e2e;
                border-radius: 5px; font-weight: bold; padding: 4px 12px;
            }
            QPushButton:hover { background: #b4befe; }
            QPushButton:pressed { background: #74c7ec; }
            QTabBar::tab {
                background: #313244; color: #a6adc8;
                border-radius: 4px 4px 0 0; padding: 5px 14px;
            }
            QTabBar::tab:selected { background: #45475a; color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
        """)
        plt.rcParams.update({
            "figure.facecolor": "#1e1e2e",
            "axes.facecolor":   "#181825",
            "axes.edgecolor":   "#45475a",
            "axes.labelcolor":  "#cdd6f4",
            "xtick.color":      "#a6adc8",
            "ytick.color":      "#a6adc8",
            "text.color":       "#cdd6f4",
            "grid.color":       "#313244",
            "grid.linestyle":   "--",
            "grid.alpha":       0.6,
            "lines.linewidth":  2.0,
        })

    def _on_material_change(self, name):
        pass

    def _get_model(self) -> PiezoModel:
        mat = MATERIALS[self.cb_material.currentText()]
        a  = self.w_a.value
        h  = self.w_h.value
        v0 = self.w_v0.value
        return PiezoModel(mat, a, h, v0)

    def on_compute(self):
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage("Вычисление…")
        QApplication.processEvents()
        try:
            model = self._get_model()
            omega = 2.0 * np.pi * self.w_freq.value
            a = model.a
            h = model.h

            r = np.linspace(0.0, a, 300)
            r[0] = 1e-12

            # смещение
            kap = model.kappa(omega)
            dlt = model.delta_det(kap)
            u = model.radial_displacement(r, omega)

            # ресет
            self.lbl_kappa.setText(f"κ = {kap:.4f} рад/м")
            self.lbl_delta.setText(f"Δ(κa) = {dlt:.4e}")
            u_max_val = np.max(np.abs(u))
            self.lbl_u_max.setText(f"u_max = {u_max_val:.4e} м")

            # амплитудно-частотная характеристика
            fmin = self.w_fmin.value
            fmax = self.w_fmax.value
            freqs, u_freq = model.frequency_response(fmin, fmax, n_pts=600)

            # резонансы
            res_list = model.find_resonances(fmin, fmax, n_pts=4000)
            if res_list:
                res_str = ", ".join(f"{f/1000:.2f} кГц" for f in res_list[:6])
                self.lbl_res.setText(f"Резонансы:\n{res_str}")
            else:
                self.lbl_res.setText("Резонансы: не найдены")

            self._model = model
            self._r = r
            self._u = u
            self._omega = omega
            self._freqs = freqs
            self._u_freq = u_freq
            self._res_list = res_list

            # рассчитываем напряжение
            c11s = model.c11s
            c12s = model.c12s
            e31s = model.e31s
            v0 = model.v0
            A = 2.0 * v0 * model.a * e31s / (h * model.delta_det(kap))  # (С учетом исправления знака ниже!)
            j1_over_r = np.zeros_like(r)
            j1_over_r[1:] = j1(kap * r[1:]) / r[1:]
            j1_over_r[0] = kap / 2.0
            du_dr = A * (kap * j0(kap * r) - j1_over_r)
            u_r   = u / r
            sigma_rr = c11s * du_dr + c12s * u_r + e31s * 2.0 * v0 / h

            self._sigma_rr = sigma_rr

            # первая вкладка
            ax = self.canvas1.axes[0]
            ax.cla()
            ax.plot(r * 1e3, u * 1e9, color="#89b4fa", label=f"f = {self.w_freq.value:.0f} Гц")
            ax.axhline(0, color="#45475a", lw=0.8)
            ax.set_xlabel("r, мм")
            ax.set_ylabel("u_r, нм")
            ax.set_title("Радиальное смещение u(r)")
            ax.legend()
            ax.grid(True)
            self.canvas1.fig.tight_layout()
            self.canvas1.draw()

            # вторая вкладка
            ax2 = self.canvas2.axes[0]
            ax2.cla()
            ax2.semilogy(freqs / 1e3, u_freq * 1e9, color="#a6e3a1", lw=1.8)
            for fr in res_list:
                ax2.axvline(fr / 1e3, color="#f38ba8", lw=1.0, ls="--", alpha=0.8)
            ax2.set_xlabel("Частота f, кГц")
            ax2.set_ylabel("|u(0.5a)|, нм")
            ax2.set_title("Амплитудно-частотная характеристика")
            ax2.grid(True)
            f_cur = self.w_freq.value
            if fmin <= f_cur <= fmax:
                ax2.axvline(f_cur / 1e3, color="#cba6f7", lw=1.5, label=f"f раб = {f_cur/1e3:.1f} кГц")
                ax2.legend()
            self.canvas2.fig.tight_layout()
            self.canvas2.draw()

            # третья вкладка
            self._replot_phi()

            # четвертая вкладка
            self._replot_phi_map()

            # пятая вкладка
            ax5 = self.canvas5.axes[0]
            ax5.cla()
            ax5.plot(r * 1e3, sigma_rr / 1e6, color="#fab387", label="σ_rr(r)")
            ax5.axhline(0, color="#45475a", lw=0.8)
            ax5.set_xlabel("r, мм")
            ax5.set_ylabel("σ_rr, МПа")
            ax5.set_title("Радиальное напряжение σ_rr(r)")
            ax5.legend()
            ax5.grid(True)
            self.canvas5.fig.tight_layout()
            self.canvas5.draw()

            # шестая вкладка
            self._replot_shape()

            if status_bar is not None:
                status_bar.showMessage("Расчёт завершён успешно")

        except (ValueError, TypeError, ZeroDivisionError) as e:
            if status_bar is not None:
                status_bar.showMessage(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def _replot_phi(self):
        if self._model is None:
            return
        model = self._model
        r = self._r
        omega = self._omega
        h = model.h
        z_norm = self.w_z_norm.value()
        z = z_norm * h / 2.0
        n_val = self.spin_nterms.value()

        ax3 = self.canvas3.axes[0]
        ax3.cla()
        phi = model.electric_potential(r, z, omega, n_terms=n_val)
        ax3.plot(r * 1e3, phi, color="#f9e2af",
                 label=f"z = {z_norm:.2f}·h/2  (z = {z*1e3:.3f} мм)")
        ax3.axhline(0, color="#45475a", lw=0.8)
        ax3.set_xlabel("r, мм")
        ax3.set_ylabel("φ, В")
        ax3.set_title("Электрический потенциал φ(r, z)")
        ax3.legend()
        ax3.grid(True)
        self.canvas3.fig.tight_layout()
        self.canvas3.draw()

    def _replot_phi_map(self):
        if self._model is None:
            return
        model = self._model
        omega = self._omega
        h = model.h
        a = model.a
        n_val = self.spin_nterms.value()

        nr, nz = 60, 40
        r_arr = np.linspace(1e-9, a, nr)
        z_arr = np.linspace(-h / 2, h / 2, nz)
        phi_matrix = np.zeros((nz, nr))
        for j, zv in enumerate(z_arr):
            phi_matrix[j, :] = model.electric_potential(r_arr, zv, omega, n_terms=n_val)

        self.canvas4.fig.clear()
        ax4 = self.canvas4.fig.add_subplot(111)
        cf = ax4.contourf(r_arr * 1e3, z_arr * 1e3, phi_matrix, levels=30, cmap="coolwarm")
        self.canvas4.fig.colorbar(cf, ax=ax4, label="φ, В")
        ax4.set_xlabel("r, мм")
        ax4.set_ylabel("z, мм")
        ax4.set_title("Распределение электрического потенциала φ(r, z)")
        self.canvas4.fig.tight_layout()
        self.canvas4.draw()

    def _replot_shape(self):
        if self._model is None:
            return
        r = self._r
        u = self._u
        ax6 = self.canvas6.axes[0]
        ax6.cla()
        amplify = 1e6  # м → мкм
        ax6.plot(r * 1e3, np.zeros_like(r), color="#45475a", lw=1, ls="--")
        ax6.plot(r * 1e3, u * amplify, color="#89dceb", lw=2, label="u(r) [мкм, амплитуда]")
        ax6.fill_between(r * 1e3, u * amplify, alpha=0.15, color="#89dceb")
        ax6.axhline(0, color="#45475a", lw=0.8)
        ax6.set_xlabel("r, мм")
        ax6.set_ylabel("u_r, мкм")
        ax6.set_title("Форма радиальных колебаний (амплитуда)")
        ax6.legend()
        ax6.grid(True)
        self.canvas6.fig.tight_layout()
        self.canvas6.draw()

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить графики", "piezo_results.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        if self._model is None:
            return
        model = self._model
        r = self._r
        u = self._u
        omega = self._omega
        h = model.h
        a = model.a

        fig, axs = plt.subplots(2, 2, figsize=(14, 9),
                                facecolor="#1e1e2e", constrained_layout=True)
        fig.suptitle(
            f"Пьезоэлектрический юниморф | {self.cb_material.currentText()} | "
            f"a={a*1e3:.1f}мм  h={h*1e3:.2f}мм  V₀={model.v0}В  f={self.w_freq.value/1e3:.2f}кГц",
            color="#cdd6f4", fontsize=11)

        # 1
        ax = axs[0, 0]
        ax.set_facecolor("#181825")
        ax.plot(r * 1e3, u * 1e9, color="#89b4fa")
        ax.set_xlabel("r, мм")
        ax.set_ylabel("u_r, нм")
        ax.set_title("Радиальное смещение u(r)")
        ax.grid(True)

        # 2
        ax = axs[0, 1]
        ax.set_facecolor("#181825")
        ax.semilogy(self._freqs / 1e3, self._u_freq * 1e9, color="#a6e3a1")
        for fr in self._res_list:
            ax.axvline(fr / 1e3, color="#f38ba8", lw=0.9, ls="--")
        ax.set_xlabel("f, кГц")
        ax.set_ylabel("|u|, нм")
        ax.set_title("АЧХ")
        ax.grid(True)

        # 3
        ax = axs[1, 0]
        ax.set_facecolor("#181825")
        phi = model.electric_potential(r, h / 4, omega)
        ax.plot(r * 1e3, phi, color="#f9e2af")
        ax.set_xlabel("r, мм")
        ax.set_ylabel("φ, В")
        ax.set_title("Потенциал φ(r, z=h/4)")
        ax.grid(True)

        # 4
        ax = axs[1, 1]
        ax.set_facecolor("#181825")
        ax.plot(r * 1e3, self._sigma_rr / 1e6, color="#fab387")
        ax.set_xlabel("r, мм")
        ax.set_ylabel("σ_rr, МПа")
        ax.set_title("Радиальное напряжение σ_rr(r)")
        ax.grid(True)

        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
        plt.close(fig)

        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Сохранено: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())