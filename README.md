# Metaanálisis del desempeño de la predicción espacial de variables ambientales mediante geoestadística y aprendizaje automático en regiones montañosas

Revisión sistemática y metaanálisis del desempeño predictivo de los métodos geoestadísticos frente al aprendizaje automático (AA) para estimar precipitación y temperatura del aire en sitios sin medición en regiones montañosas (2020–2026), siguiendo PRISMA 2020.

**Universidad Nacional del Altiplano** · Facultad de Ingeniería Estadística e Informática
Curso: Estadística Espacial · Docente: Fred Torres Cruz

**Autores:** Kris Ximena Colquehuanca Zapana · Yeferson Dariun Laura Livise · Obed Eliu Leon Mamani · Kevin Pleyer Portillo Machaca

## Resultado principal

- 18 estudios incluidos; el metaanálisis principal combina 16 comparaciones de 15 estudios.
- En promedio, el AA tiene un RMSE **12,9 % menor** (IC 95 %: 4,5–20,6 %; ln-razón = 0,138; p = 0,006), con heterogeneidad moderada (I² = 60 %).
- La ventaja depende de las covariables disponibles y del esquema de validación, y **desaparece en los estudios con bajo riesgo de sesgo** (ln-razón = −0,022).

## Estructura

| Carpeta | Contenido |
|---|---|
| `01_protocolo/` | PICO, propuesta y criterios de inclusión y exclusión |
| `02_busqueda_seleccion/` | Recolección de registros, informes de búsqueda y diagrama PRISMA |
| `03_extraccion/` | Ficha de extracción v4 (Excel y Word), matriz de los estudios incluidos e informe consolidado por integrante |
| `articulo_ieee/` | Artículo en formato IEEE (`articulo.tex`, `articulo.pdf`), presentación y figuras |
| `articulo_ieee/analisis/` | Datos del metaanálisis (`datos_extraccion.csv`), código (`metaanalisis.py`) y resultados |
| `archivo/` | Versiones anteriores de la ficha y de la presentación |

Los PDF de los artículos analizados no se incluyen porque tienen derechos de autor; sus referencias completas (con DOI) están en el artículo y en la ficha de extracción.

## Reproducir el metaanálisis

Requiere Python 3 con `numpy`, `scipy` y `matplotlib`.

```bash
cd articulo_ieee/analisis
python metaanalisis.py
```

El script lee `datos_extraccion.csv`, ajusta un modelo de efectos aleatorios (REML con ajuste de Hartung–Knapp) sobre ln(RMSE espacial / RMSE AA), escribe `resultados/resultados.json` y regenera los gráficos de bosque y de embudo en `articulo_ieee/figuras/`.

Para compilar el artículo:

```bash
cd articulo_ieee
pdflatex articulo.tex
pdflatex articulo.tex
```
