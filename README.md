# FMCG Revealed Preferences Simulator

Aplicación en Streamlit para simular y analizar preferencias reveladas a partir de variables típicas de revenue management en FMCG:

- precio
- descuento
- distribución
- display
- equity de marca
- estacionalidad

## Qué hace

1. Genera datos sintéticos de SKU-mercado-semana o carga un CSV base.
2. Simula comportamiento observado de elección.
3. Estima una aproximación de preferencia revelada.
4. Permite correr escenarios de price / promo / distribución.

## Estructura

- `app.py`: aplicación principal
- `requirements.txt`: dependencias
- `sample_data.csv`: ejemplo de datos
- `.gitignore`: exclusiones estándar
- `LICENSE`: licencia del repositorio

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
