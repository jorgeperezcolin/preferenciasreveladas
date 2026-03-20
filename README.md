# FMCG Revealed Preferences Simulator

Aplicación en Streamlit para simular y analizar preferencias reveladas a partir de variables típicas de revenue management en FMCG.

## Qué hace este proyecto

Este repositorio implementa un simulador que integra tres capas analíticas:

1. **Simulación de comportamiento de compra**
   - Genera datos SKU–mercado–semana
   - Modela decisiones del consumidor con utilidad latente

2. **Estimación de preferencias reveladas**
   - Aproxima la utilidad a partir de comportamiento observado
   - Identifica drivers clave: precio, promoción, distribución, marca

3. **Simulación de escenarios (Revenue Management)**
   - Cambios en precio
   - Cambios en promoción
   - Cambios en distribución y exhibición

4. **Curvas de indiferencia**
   - Representación microeconómica de trade-offs
   - Relación entre precio y valor de marca/calidad
   - Interpretación por segmento de consumidor

---

## Arquitectura del repositorio

```text
fmcg-revealed-preferences/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── sample_data.csv
├── smoke_test.py
└── src/
    ├── __init__.py
    ├── data_generator.py
    ├── preference_model.py
    ├── scenario_engine.py
    └── indifference.py
