import itertools
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import html
from scipy.optimize import linprog


# ================================================================
# CONFIGURACIÓN
# ================================================================

st.set_page_config(
    page_title="Calculadora de programación lineal",
    page_icon="📈",
    layout="wide"
)

TOL = 1e-8


# ================================================================
# DISEÑO
# ================================================================

st.markdown("""
<style>

/* Fondo general y texto */
.stApp {
    background-color: #F4F7FB;
    color: #243247;
}

/* Títulos y encabezados */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {
    color: #17365D !important;
    font-weight: 700 !important;
}

/* Etiquetas de los campos */
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #26384D !important;
}

/* Formularios */
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #D6E0EC;
    box-shadow: 0 2px 8px rgba(30, 55, 85, 0.05);
}

/* Campos de entrada */
.stApp input,
.stApp textarea {
    color: #1F2937 !important;
}

/* Botón de calcular */
.stFormSubmitButton > button {
    background-color: #315B91 !important;
    color: #FFFFFF !important;
    border-radius: 10px;
    border: none !important;
    font-weight: bold;
    min-height: 44px;
}

.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

/* Desplegable del procedimiento */
div[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #CCD9E8;
    border-radius: 12px;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
    color: #17365D !important;
    font-weight: 700 !important;
}

/* Recuadro del procedimiento */
.procedimiento {
    background-color: #FFFFFF;
    color: #243247;
    border: 1px solid #D6E0EC;
    border-radius: 10px;
    padding: 20px;
    overflow-x: auto;
}

.procedimiento pre {
    color: #243247 !important;
    background-color: #FFFFFF !important;
    font-family: Consolas, "Courier New", monospace;
    font-size: 14px;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin: 0;
}

/* Adaptación a móviles */
@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }

    div[data-testid="stHorizontalBlock"] > div {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    div[data-testid="stForm"] {
        padding: 14px;
    }

    .procedimiento {
        padding: 12px;
    }

    .procedimiento pre {
        font-size: 12px;
    }
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# FUNCIONES AUXILIARES
# ================================================================

def fmt_num(v):
    if abs(v) < 1e-10:
        v = 0.0
    return f"{v:g}"


def texto_restriccion(r):
    a, b, signo, c = r
    return f"{fmt_num(a)}x {b:+g}y {signo} {fmt_num(c)}"


def es_estricta(signo):
    return signo in ("<", ">")


def nombre_vertice(i):
    if i <= 26:
        return chr(64 + i)
    return f"A{i-26}"


def satisface_cerradas(p, restricciones, tol=1e-7):
    x, y = p

    if x < -tol or y < -tol:
        return False

    for a, b, signo, c in restricciones:
        v = a*x + b*y

        if signo in ("<=", "<") and v > c + tol:
            return False

        if signo in (">=", ">") and v < c - tol:
            return False

        if signo == "=" and abs(v-c) > tol:
            return False

    return True


def satisface_reales(p, restricciones, tol=1e-7):
    x, y = p

    if x < -tol or y < -tol:
        return False

    for a, b, signo, c in restricciones:
        v = a*x + b*y

        if signo == "<=" and v > c + tol:
            return False

        if signo == "<" and v >= c - tol:
            return False

        if signo == ">=" and v < c - tol:
            return False

        if signo == ">" and v <= c + tol:
            return False

        if signo == "=" and abs(v-c) > tol:
            return False

    return True


# ================================================================
# MODELO DE PROGRAMACIÓN LINEAL
# ================================================================

def matrices_lp(restricciones):
    Aub, bub, Aeq, beq = [], [], [], []

    for a, b, signo, c in restricciones:

        if signo in ("<=", "<"):
            Aub.append([a, b])
            bub.append(c)

        elif signo in (">=", ">"):
            Aub.append([-a, -b])
            bub.append(-c)

        else:
            Aeq.append([a, b])
            beq.append(c)

    return (
        np.array(Aub, dtype=float).reshape((-1, 2)),
        np.array(bub, dtype=float),
        np.array(Aeq, dtype=float).reshape((-1, 2)),
        np.array(beq, dtype=float)
    )


def resolver_lp(objetivo, maximizar, restricciones, extra_eq=None):
    Aub, bub, Aeq, beq = matrices_lp(restricciones)

    if extra_eq is not None:
        fila, valor = extra_eq
        Aeq = np.vstack([Aeq, np.asarray(fila, dtype=float)])
        beq = np.append(beq, valor)

    c = -objetivo if maximizar else objetivo

    return linprog(
        c,
        A_ub=Aub if len(Aub) else None,
        b_ub=bub if len(bub) else None,
        A_eq=Aeq if len(Aeq) else None,
        b_eq=beq if len(beq) else None,
        bounds=[(0, None), (0, None)],
        method="highs"
    )


# ================================================================
# DESIGUALDADES ESTRICTAS
# ================================================================

def comprobar_factibilidad_estricta(restricciones):
    estrictas = [r for r in restricciones if es_estricta(r[2])]

    if not estrictas:
        res = resolver_lp(
            np.array([0., 0.]),
            False,
            restricciones
        )
        return res.success, None, res

    Aub, bub, Aeq, beq = matrices_lp(restricciones)

    filas = []

    for a, b, signo, c in restricciones:
        if signo == "<":
            filas.append([a, b, 1.])
        elif signo == ">":
            filas.append([-a, -b, 1.])

    Aub3 = (
        np.column_stack([Aub, np.zeros(len(Aub))])
        if len(Aub) else np.empty((0, 3))
    )

    bub3 = bub.copy()

    if filas:
        Aub3 = np.vstack([Aub3, np.array(filas, dtype=float)])

        bub3 = np.concatenate([
            bub3,
            np.array([
                r[3] if r[2] == "<" else -r[3]
                for r in estrictas
            ], dtype=float)
        ])

    Aeq3 = (
        np.column_stack([Aeq, np.zeros(len(Aeq))])
        if len(Aeq) else np.empty((0, 3))
    )

    res = linprog(
        [0., 0., -1.],
        A_ub=Aub3 if len(Aub3) else None,
        b_ub=bub3 if len(bub3) else None,
        A_eq=Aeq3 if len(Aeq3) else None,
        b_eq=beq if len(beq) else None,
        bounds=[(0, None), (0, None), (0, None)],
        method="highs"
    )

    if res.status == 3:
        return True, np.inf, res

    if not res.success:
        return False, None, res

    return res.x[2] > 1e-8, res.x[2], res


def comprobar_factibilidad_estricta_en_cara(
    restricciones, objetivo, z_opt
):
    estrictas = [r for r in restricciones if es_estricta(r[2])]

    Aub, bub, Aeq, beq = matrices_lp(restricciones)

    filas = []

    for a, b, signo, c in restricciones:
        if signo == "<":
            filas.append([a, b, 1.])
        elif signo == ">":
            filas.append([-a, -b, 1.])

    Aub3 = (
        np.column_stack([Aub, np.zeros(len(Aub))])
        if len(Aub) else np.empty((0, 3))
    )

    bub3 = bub.copy()

    if filas:
        rhs = [
            r[3] if r[2] == "<" else -r[3]
            for r in estrictas
        ]

        Aub3 = np.vstack([
            Aub3,
            np.array(filas, dtype=float)
        ])

        bub3 = np.concatenate([
            bub3,
            np.array(rhs, dtype=float)
        ])

    Aeq3 = (
        np.column_stack([Aeq, np.zeros(len(Aeq))])
        if len(Aeq) else np.empty((0, 3))
    )

    Aeq3 = np.vstack([
        Aeq3,
        np.array([objetivo[0], objetivo[1], 0.])
    ])

    beq3 = np.append(beq, z_opt)

    res = linprog(
        [0., 0., -1.],
        A_ub=Aub3 if len(Aub3) else None,
        b_ub=bub3 if len(bub3) else None,
        A_eq=Aeq3,
        b_eq=beq3,
        bounds=[(0, None), (0, None), (0, None)],
        method="highs"
    )

    if res.status == 3:
        return True, np.inf, res

    if not res.success:
        return False, None, res

    return res.x[2] > 1e-8, res.x[2], res


def punto_optimo_alcanzable(objetivo, maximizar, restricciones, z_opt):
    if not any(es_estricta(r[2]) for r in restricciones):
        return True, None, None

    return comprobar_factibilidad_estricta_en_cara(
        restricciones, objetivo, z_opt
    )


# ================================================================
# CÁLCULO DE VÉRTICES
# ================================================================

def obtener_vertices_cierre(restricciones):
    fronteras = [(a, b, c) for a, b, signo, c in restricciones]

    fronteras += [
        (1., 0., 0.),
        (0., 1., 0.)
    ]

    puntos = []

    for r1, r2 in itertools.combinations(fronteras, 2):
        M = np.array([
            [r1[0], r1[1]],
            [r2[0], r2[1]]
        ], dtype=float)

        if abs(np.linalg.det(M)) < 1e-10:
            continue

        p = np.linalg.solve(
            M,
            np.array([r1[2], r2[2]], dtype=float)
        )

        if satisface_cerradas(p, restricciones):
            p[np.abs(p) < 1e-9] = 0.

            if not any(np.linalg.norm(p-q) < 1e-6 for q in puntos):
                puntos.append(p)

    return sorted(puntos, key=lambda p: (p[0], p[1]))


# ================================================================
# TABLAS DE PUNTOS PARA REPRESENTAR LAS RECTAS
# ================================================================

def tabla_puntos_rectas(restricciones):
    print("\nPUNTOS PARA REPRESENTAR CADA RECTA")

    tablas = []

    for i, (a, b, signo, c) in enumerate(restricciones, 1):
        print(f"\nR{i}: {texto_restriccion((a, b, signo, c))}")

        puntos = []

        def agregar(x, y):
            if not np.isfinite(x) or not np.isfinite(y):
                return

            if x < -1e-8 or y < -1e-8:
                return

            x = 0.0 if abs(x) < 1e-8 else float(x)
            y = 0.0 if abs(y) < 1e-8 else float(y)

            if not any(
                abs(x-p[0]) < 1e-7 and abs(y-p[1]) < 1e-7
                for p in puntos
            ):
                puntos.append((x, y))

        # Buscamos primero puntos enteros.
        if abs(b) <= TOL and abs(a) > TOL:
            x = c/a
            if abs(x-round(x)) < 1e-8:
                for y in range(21):
                    agregar(round(x), y)

        elif abs(b) > TOL:
            for x in range(21):
                y = (c-a*x)/b

                if abs(y-round(y)) < 1e-8:
                    agregar(x, round(y))

                if len(puntos) >= 3:
                    break

        # Si faltan puntos, probamos valores decimales sencillos.
        if len(puntos) < 3:
            valores_x = (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)

            if abs(b) > TOL:
                for x in valores_x:
                    agregar(x, (c-a*x)/b)
                    if len(puntos) >= 3:
                        break

            elif abs(a) > TOL:
                x = c/a
                for y in (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5):
                    agregar(x, y)
                    if len(puntos) >= 3:
                        break

        if puntos:
            df = pd.DataFrame([
                {"x": x, "y": y}
                for x, y in puntos[:3]
            ])

            tablas.append((
                f"R{i}",
                df.style.format({"x": "{:.5g}", "y": "{:.5g}"})
            ))
        else:
            print("No se encuentran puntos de esta recta en el primer cuadrante.")

    if tablas:
        display(tablas)


# ================================================================
# PROCEDIMIENTO DE CÁLCULO DE VÉRTICES
# ================================================================

def mostrar_procedimiento_vertices(restricciones, vertices):
    print("\nCÁLCULO DE INTERSECCIONES DE LAS RECTAS")

    fronteras = [
        (a, b, c, f"R{i}")
        for i, (a, b, signo, c) in enumerate(restricciones, 1)
    ]

    fronteras += [
        (1., 0., 0., "Eje Y (x=0)"),
        (0., 1., 0., "Eje X (y=0)")
    ]

    if not vertices:
        print("No hay vértices en el cierre de la región factible.")
        return

    mostradas = set()

    for (a1, b1, c1, n1), (a2, b2, c2, n2) in itertools.combinations(fronteras, 2):
        M = np.array([
            [a1, b1],
            [a2, b2]
        ], dtype=float)

        if abs(np.linalg.det(M)) < 1e-10:
            continue

        p = np.linalg.solve(M, np.array([c1, c2], dtype=float))

        if not satisface_cerradas(p, restricciones):
            continue

        p[np.abs(p) < 1e-9] = 0.

        idx = next(
            (i for i, v in enumerate(vertices, 1)
             if np.linalg.norm(v-p) < 1e-6),
            None
        )

        if idx is None or idx in mostradas:
            continue

        mostradas.add(idx)
        letra = nombre_vertice(idx)

        print(f"\n{letra}: intersección de {n1} y {n2}")
        print("Sistema de ecuaciones:")
        print(f"  {a1:g}x {b1:+g}y = {c1:g}")
        print(f"  {a2:g}x {b2:+g}y = {c2:g}")

        if abs(a1) > TOL and abs(a2) > TOL:
            m1, m2 = a2, -a1
            B = m1*b1 + m2*b2
            C = m1*c1 + m2*c2

            print("\nMultiplicamos las ecuaciones:")
            print(f"  E1 por {m1:g}: {m1*a1:g}x {m1*b1:+g}y = {m1*c1:g}")
            print(f"  E2 por {m2:g}: {m2*a2:g}x {m2*b2:+g}y = {m2*c2:g}")
            print("\nSumamos las ecuaciones:")

            if abs(B) > TOL:
                print(f"  {B:g}y = {C:g}")
                print(f"  y = {C:g} / {B:g} = {p[1]:.6g}")
                print("\nSustituimos en la primera ecuación:")
                print(f"  {a1:g}x + ({b1:g})·({p[1]:.6g}) = {c1:g}")
                print(f"  x = {p[0]:.6g}")
            else:
                print(f"  {B:g}y = {C:g}")
                print(f"  x = {p[0]:.6g}")
                print(f"  y = {p[1]:.6g}")
        else:
            print(f"Resolviendo el sistema: x = {p[0]:.6g}, y = {p[1]:.6g}")

        print(f"\nResultado: {letra} = ({p[0]:.6g}, {p[1]:.6g})")


# ================================================================
# GRÁFICA
# ================================================================

def limites_grafica(vertices, restricciones):
    xs = [p[0] for p in vertices if p[0] >= 0]
    ys = [p[1] for p in vertices if p[1] >= 0]

    for a, b, signo, c in restricciones:
        if abs(a) > TOL and c/a >= 0:
            xs.append(c/a)

        if abs(b) > TOL and c/b >= 0:
            ys.append(c/b)

    xmax = min(max(10., max(xs, default=0)*1.3+1), 5000)
    ymax = min(max(10., max(ys, default=0)*1.3+1), 5000)

    return xmax, ymax


def dibujar_region(
    restricciones,
    vertices,
    puntos_optimos=None,
    etiqueta_optimo=None
):
    xmax, ymax = limites_grafica(vertices, restricciones)

    xs = np.linspace(0, xmax, 1000)
    ys = np.linspace(0, ymax, 1000)

    X, Y = np.meshgrid(xs, ys)
    factible = np.ones_like(X, dtype=bool)

    for a, b, signo, c in restricciones:
        v = a*X + b*Y

        if signo in ("<=", "<"):
            factible &= v <= c + 1e-9

        elif signo in (">=", ">"):
            factible &= v >= c - 1e-9

        else:
            factible &= np.abs(v-c) <= max(xmax, ymax)/1000

    # Tamaño moderado para la pantalla.
    # El archivo de imagen se guarda con 300 ppp.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)

    ax.set_facecolor("#fbfdff")

    if np.any(factible):
        ax.contourf(
            X, Y, factible.astype(int),
            levels=[0.5, 1.5],
            colors=["#8ecae6"],
            alpha=0.58,
            antialiased=True
        )

    for i, (a, b, signo, c) in enumerate(restricciones, 1):
        estilo = "--" if signo in ("<", ">") else "-"
        etiqueta = f"R{i}: {texto_restriccion((a, b, signo, c))}"

        if abs(b) > TOL:
            yy = (c-a*xs)/b
            mask = (yy >= 0) & (yy <= ymax)

            if np.any(mask):
                ax.plot(
                    xs[mask], yy[mask],
                    linestyle=estilo,
                    linewidth=2.2,
                    label=etiqueta,
                    zorder=3
                )

        elif abs(a) > TOL:
            xx = c/a

            if 0 <= xx <= xmax:
                ax.axvline(
                    xx,
                    linestyle=estilo,
                    linewidth=2.2,
                    label=etiqueta,
                    zorder=3
                )

    ax.axhline(0, color="black", linewidth=1.3)
    ax.axvline(0, color="black", linewidth=1.3)

    if vertices:
        desplazamientos = [
            (12, 14), (12, -27), (-125, 14),
            (-125, -27), (14, 34), (-120, 34)
        ]

        for i, p in enumerate(vertices, 1):
            incluido = satisface_reales(p, restricciones)

            if incluido:
                ax.scatter(
                    [p[0]], [p[1]], s=105,
                    color="crimson",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=6
                )
            else:
                ax.scatter(
                    [p[0]], [p[1]], s=115,
                    facecolors="white",
                    edgecolors="crimson",
                    linewidth=2,
                    zorder=6
                )

            dx, dy = desplazamientos[(i-1) % len(desplazamientos)]

            ax.annotate(
                f"{nombre_vertice(i)} ({p[0]:.3g}, {p[1]:.3g})",
                (p[0], p[1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=.25",
                    fc="white",
                    ec="#777777",
                    alpha=.95
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#555555",
                    lw=0.8
                ),
                zorder=7
            )

    if puntos_optimos:
        for j, (p, letra) in enumerate(puntos_optimos):
            ax.scatter(
                [p[0]], [p[1]],
                marker="*",
                s=350,
                color="gold",
                edgecolors="darkorange",
                linewidths=1.5,
                zorder=10,
                label="Óptimo" if j == 0 else None
            )

        p = puntos_optimos[0][0]

        ax.annotate(
            etiqueta_optimo or "Óptimo",
            (p[0], p[1]),
            xytext=(20, 35),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="darkred",
            bbox=dict(
                boxstyle="round,pad=.4",
                fc="#fff3cd",
                ec="darkorange",
                alpha=.98
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="darkorange",
                lw=1.5
            ),
            zorder=11
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.set_title("Región factible y vértices", fontsize=15, pad=12)

    ax.grid(True, which="major", alpha=.25, linewidth=.8)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)

    if restricciones:
        ax.legend(loc="best", fontsize=9, framealpha=.95)

    ax.text(
        .015, .985,
        "● Vértice incluido     ○ Vértice excluido por desigualdad estricta",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=.3",
            fc="white",
            ec="#aaaaaa",
            alpha=.9
        )
    )

    fig.tight_layout()

    nombre = "grafica_programacion_lineal.png"
    fig.savefig(nombre, dpi=300, bbox_inches="tight")

    # Gráfica centrada.
    col_izq, col_grafica, col_der = st.columns([1, 4, 1])

    with col_grafica:
        st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    print(f"\nGráfica guardada como {nombre} (300 ppp).")
    print("Las rectas discontinuas corresponden a desigualdades estrictas (< o >).")


# ================================================================
# RESOLUCIÓN COMPLETA
# ================================================================

def resolver(objetivo, maximizar, restricciones):

    print("\n" + "="*72)
    print("1. PLANTEAMIENTO DEL PROBLEMA")
    print("="*72)

    print(f"Función objetivo: Z = {objetivo[0]:g}x + {objetivo[1]:g}y")
    print("Se pide:", "MAXIMIZAR" if maximizar else "MINIMIZAR")

    print("\nRestricciones:")

    if restricciones:
        for i, r in enumerate(restricciones, 1):
            marca = (
                " (estricta: frontera discontinua)"
                if es_estricta(r[2]) else ""
            )
            print(f"  R{i}: {texto_restriccion(r)}{marca}")
    else:
        print("  Ninguna restricción adicional.")

    print("No negatividad: x >= 0, y >= 0.")

    print("\n" + "="*72)
    print("2. TABLAS DE PUNTOS PARA DIBUJAR LAS RECTAS")
    print("="*72)

    tabla_puntos_rectas(restricciones)

    print("\n" + "="*72)
    print("3. CÁLCULO DE LOS VÉRTICES")
    print("="*72)

    vertices = obtener_vertices_cierre(restricciones)
    mostrar_procedimiento_vertices(restricciones, vertices)

    # Dos columnas: una tabla de vértices y otra de valores de Z.
    col_vertices, col_objetivo = st.columns(2, gap="large")

    # ------------------------------------------------------------
    # TABLA 1: VÉRTICES
    # ------------------------------------------------------------

    with col_vertices:

        st.markdown("### Vértices")

        if vertices:

            print("\nListado de vértices del cierre de la región:")

            filas = []

            for i, p in enumerate(vertices, 1):

                incluido = satisface_reales(p, restricciones)

                filas.append({
                    "Vértice": nombre_vertice(i),
                    "x": p[0],
                    "y": p[1],
                    "¿Pertenece?": (
                        "Incluido" if incluido
                        else "Excluido (frontera estricta)"
                    )
                })

            display(pd.DataFrame(filas).style.format({
                "x": "{:.5g}",
                "y": "{:.5g}"
            }))

        else:

            print("No se han encontrado vértices.")
            st.info("No se han encontrado vértices.")

    # ------------------------------------------------------------
    # TABLA 2: FUNCIÓN OBJETIVO
    # ------------------------------------------------------------

    print("\n" + "="*72)
    print("4. TABLA DE LA FUNCIÓN OBJETIVO")
    print("="*72)

    with col_objetivo:

        st.markdown("### Valores de la función objetivo")

        if vertices:

            datos = []

            for i, p in enumerate(vertices, 1):

                datos.append({
                    "Vértice": nombre_vertice(i),
                    "x": p[0],
                    "y": p[1],
                    "Z = a·x + b·y": float(objetivo @ p),
                    "Punto admisible": (
                        "Sí" if satisface_reales(p, restricciones)
                        else "No"
                    )
                })

            display(pd.DataFrame(datos).style.format({
                "x": "{:.5g}",
                "y": "{:.5g}",
                "Z = a·x + b·y": "{:.5g}"
            }))

        else:

            print("No hay vértices que incluir en la tabla.")
            st.info("No hay vértices que incluir en la tabla.")
    print("5. OPTIMIZACIÓN Y CONCLUSIÓN")
    print("="*72)

    puntos_optimos_grafica = []
    etiqueta_optimo_grafica = None

    factible_estricto, _, _ = comprobar_factibilidad_estricta(
        restricciones
    )

    if not factible_estricto:
        print(
            "El problema no tiene puntos factibles: "
            "las restricciones son incompatibles."
        )

    else:
        res = resolver_lp(objetivo, maximizar, restricciones)

        if res.status == 3:
            print(
                "La región factible es no acotada en el sentido "
                "de optimización elegido."
            )
            print("No existe un máximo/mínimo finito.")

        elif res.status == 2:
            print("No existe solución factible.")

        elif res.success:
            punto = res.x
            z = float(objetivo @ punto)

            alcanzado, _, _ = punto_optimo_alcanzable(
                objetivo,
                maximizar,
                restricciones,
                z
            )

            nombre = "máximo" if maximizar else "mínimo"

            if alcanzado:
                print(f"El {nombre} es Z = {z:.6g}.")
                print(
                    f"Se alcanza en el punto "
                    f"({punto[0]:.6g}, {punto[1]:.6g})."
                )

                ids = [
                    i for i, p in enumerate(vertices, 1)
                    if (
                        np.linalg.norm(p-punto) < 1e-6
                        and satisface_reales(p, restricciones)
                    )
                ]

                if ids:
                    print(
                        "Vértice:",
                        ", ".join(nombre_vertice(i) for i in ids) + "."
                    )
                else:
                    print(
                        "El óptimo pertenece a la región factible; "
                        "puede estar en un lado o segmento óptimo."
                    )

                valores = [float(objetivo @ p) for p in vertices]

                opt_ids = [
                    i for i, (p, v) in enumerate(
                        zip(vertices, valores), 1
                    )
                    if (
                        abs(v-z) < 1e-6
                        and satisface_reales(p, restricciones)
                    )
                ]

                puntos_optimos_grafica = [
                    (vertices[i-1], nombre_vertice(i))
                    for i in opt_ids
                ]

                tipo_optimo = "MÁXIMO" if maximizar else "MÍNIMO"

                if opt_ids:
                    letras = ", ".join(
                        nombre_vertice(i) for i in opt_ids
                    )
                    etiqueta_optimo_grafica = (
                        f"{tipo_optimo} Z = {z:g}\nVértice(s): {letras}"
                    )
                else:
                    puntos_optimos_grafica = [(punto, "")]
                    etiqueta_optimo_grafica = f"{tipo_optimo} Z = {z:g}"

                if len(opt_ids) > 1:
                    print(
                        "También se alcanza en los vértices:",
                        ", ".join(nombre_vertice(i) for i in opt_ids) + "."
                    )

            else:
                tipo = "supremo" if maximizar else "ínfimo"

                print(
                    f"No existe un {nombre} alcanzado dentro "
                    "de la región factible."
                )
                print(f"El {tipo} es Z = {z:.6g}, pero no se alcanza")
                print(
                    "porque el punto óptimo queda excluido "
                    "por al menos una desigualdad estricta."
                )
                print(
                    f"Punto del cierre donde se obtiene ese valor: "
                    f"({punto[0]:.6g}, {punto[1]:.6g})."
                )

        else:
            print("No se ha podido resolver el problema:", res.message)

    print("\n" + "="*72)
    print("6. REPRESENTACIÓN GRÁFICA")
    print("="*72)

    dibujar_region(
        restricciones,
        vertices,
        puntos_optimos_grafica,
        etiqueta_optimo_grafica
    )


# ================================================================
# INTERFAZ
# ================================================================

st.markdown("""
<div style="
    background: linear-gradient(120deg, #23436B, #477DB8);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 22px;
">
    <h1 style="color:white; margin-bottom:5px;">
        📈 Programación lineal
    </h1>
    <h3 style="color:#E4EEFA; font-weight:normal;">
        Matemáticas Aplicadas a las CCSS II
    </h3>
    <p style="color:white; margin-bottom:0;">
        2.º de Bachillerato · Calculadora interactiva
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Introduce la función objetivo y sus restricciones. "
    "Obtendrás las tablas, la región factible, los vértices, "
    "la solución óptima y el procedimiento explicado."
)

col_num, col_vacio = st.columns([1, 3])

with col_num:
    n_restricciones = st.number_input(
        "Número de restricciones",
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )


with st.form("formulario_problema"):

    st.subheader("1. Función objetivo")

    col_x, col_y, col_tipo = st.columns([1, 1, 1.2])

    with col_x:
        coef_x = st.number_input(
            "Coeficiente de x",
            value=3.0,
            step=1.0,
            format="%g",
            key="coef_x"
        )

    with col_y:
        coef_y = st.number_input(
            "Coeficiente de y",
            value=2.0,
            step=1.0,
            format="%g",
            key="coef_y"
        )

    with col_tipo:
        tipo_opt = st.selectbox(
            "Optimizar",
            ["Máximo", "Mínimo"]
        )

    st.subheader("2. Restricciones")

    restricciones_form = []
    signos = ["<=", "<", ">=", ">", "="]

    for i in range(int(n_restricciones)):

        st.markdown(f"**Restricción {i+1}**")

        c1, c2, c3, c4 = st.columns([1, 1, 0.8, 1])

        with c1:
            aa = st.number_input(
                "Coeficiente de x",
                value=1.0,
                step=1.0,
                format="%g",
                key=f"ra_{i}"
            )

        with c2:
            bb = st.number_input(
                "Coeficiente de y",
                value=1.0,
                step=1.0,
                format="%g",
                key=f"rb_{i}"
            )

        with c3:
            ss = st.selectbox(
                "Signo",
                signos,
                key=f"rs_{i}"
            )

        with c4:
            cc = st.number_input(
                "Término independiente",
                value=10.0,
                step=1.0,
                format="%g",
                key=f"rc_{i}"
            )

        restricciones_form.append(
            (float(aa), float(bb), ss, float(cc))
        )

    enviar = st.form_submit_button(
        "Calcular solución",
        type="primary",
        use_container_width=True
    )


# ================================================================
# CONSERVAR Y MOSTRAR LOS RESULTADOS
# ================================================================

if enviar:
    st.session_state["datos_problema"] = {
        "objetivo": np.array(
            [float(coef_x), float(coef_y)],
            dtype=float
        ),
        "maximizar": tipo_opt == "Máximo",
        "restricciones": restricciones_form
    }


datos_guardados = st.session_state.get("datos_problema")


if datos_guardados is None:

    st.info(
        "Completa los datos y pulsa «Calcular solución» "
        "para obtener los resultados."
    )

else:

    st.divider()
    st.header("Resultados")

    # Muestra las tablas sin dividir las tablas de resultados.
    def display(obj):

        # Tablas de puntos para dibujar las rectas.
        # Se mantienen en tres columnas.
        if isinstance(obj, list):

            columnas = st.columns(3)

            for i, (titulo, tabla) in enumerate(obj):

                with columnas[i % 3]:

                    st.markdown(f"**{titulo}**")

                    st.dataframe(
                        tabla,
                        use_container_width=True,
                        hide_index=True
                    )

            return

        # Tablas de vértices y función objetivo.
        # Se muestran completas, sin trocearlas.
        st.dataframe(
            obj,
            use_container_width=True,
            hide_index=True
        )                    
    # Capturamos el procedimiento para mostrarlo en el desplegable.
    salida = io.StringIO()

    try:
        with st.spinner("Resolviendo el problema..."):

            with redirect_stdout(salida):
                resolver(
                    datos_guardados["objetivo"],
                    datos_guardados["maximizar"],
                    datos_guardados["restricciones"]
                )

    except Exception as error:
        st.error(
            f"Se ha producido un error al resolver el problema: {error}"
        )

    texto_procedimiento = salida.getvalue()

    if texto_procedimiento.strip():
    with st.expander(
        "📘 Ver procedimiento paso a paso",
        expanded=False
    ):
        st.markdown(
            f"""
            <div class="procedimiento">
                <pre>{html.escape(texto_procedimiento)}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )
