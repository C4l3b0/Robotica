# Proyecto Dron - Robótica

Este proyecto implementa el control de un dron DJI Tello con capacidades de detección de objetos (YOLO), seguimiento de personas y detección de AprilTags.

## Requisitos Previos

Asegúrate de tener instalado Python 3.8 o superior.

## Configuración del Proyecto

Sigue estos pasos para configurar el entorno y ejecutar el código:

### 1. Crear un Entorno Virtual
Es recomendable usar un entorno virtual para mantener las dependencias aisladas. Ejecuta el siguiente comando en la terminal:

```bash
python -m venv venv
```

### 2. Activar el Entorno Virtual

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar las Librerías Necesarias
Una vez activado el entorno virtual, instala las dependencias utilizando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Código Principal
Para iniciar la implementación principal del dron, ejecuta:

```bash
python Implementacion.py
```