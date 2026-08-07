# Módulo `utils.logger`

## Descripción General

El módulo `utils.logger` implementa el sistema de trazabilidad experimental del proyecto CICIDS2017. Proporciona un logger jerárquico de doble salida —consola y fichero— que garantiza que la ejecución completa de cualquier pipeline quede registrada de forma íntegra y reproducible en `output/results.txt`. Todos los módulos del proyecto obtienen loggers derivados del logger raíz configurado por este módulo, evitando la configuración dispersa y los efectos colaterales habituales en el módulo `logging` estándar de Python.

## Fundamento Teórico

El sistema se fundamenta en la arquitectura jerárquica del módulo `logging` de la biblioteca estándar de Python (PEP 282), que organiza los loggers en un árbol de nombres separados por puntos. Un mensaje emitido por un logger hijo se propaga automáticamente hacia el logger raíz salvo que `propagate` esté desactivado; esto permite que un único par de handlers (consola + fichero) capture la actividad de todos los submódulos sin reconfiguración.

La elección de niveles de severidad sigue el estándar RFC 5424 (Syslog):

| Nivel | Valor numérico | Semántica en este proyecto |
|-------|---------------|---------------------------|
| `DEBUG` | 10 | Valores intermedios de entrenamiento, dimensiones de tensores |
| `INFO` | 20 | Hitos de pipeline, métricas por época, tiempos de carga |
| `WARNING` | 30 | Clases eliminadas por hiper-minoría, imputación de NaN |
| `ERROR` | 40 | Fallos recuperables (fichero no encontrado, split imposible) |

El logger raíz se fija en `DEBUG` para capturar todos los niveles; la filtración se delega a cada handler: el handler de consola aplica el nivel `console_level` (por defecto `INFO`) y el handler de fichero captura desde `DEBUG` para preservar la traza completa en disco.

## Funciones

### `setup_logger(output_path: Path, console_level: int = logging.INFO) -> logging.Logger`

**Descripción:** Configura el logger raíz del proyecto con un `StreamHandler` hacia `stdout` y un `FileHandler` que escribe en `output_path/results.txt`. La función es **idempotente**: si el logger ya dispone de handlers (p. ej., por una llamada previa en el mismo proceso), retorna inmediatamente sin añadir handlers duplicados.

**Parámetros:**
- `output_path` — ruta al directorio de salida; se crea si no existe mediante `Path.mkdir(parents=True, exist_ok=True)`.
- `console_level` — nivel mínimo de severidad que el handler de consola reenvía a `stdout`. Acepta constantes de `logging` (`logging.DEBUG`, `logging.INFO`, etc.). Por defecto `logging.INFO`.

**Retorna:** El logger `logging.Logger` con nombre `"cicids2017"` configurado con ambos handlers.

**Notas:**
- Debe invocarse **una única vez**, desde `main.py`, antes de importar cualquier módulo que llame a `get_logger`.
- El fichero `results.txt` se abre en modo `"a"` (append) para conservar ejecuciones anteriores dentro de la misma sesión del proceso.
- El formato de cada línea es `YYYY-MM-DD HH:MM:SS [NIVEL   ] nombre — mensaje`, lo que facilita el grep y la auditoría posterior.

---

### `get_logger(name: str) -> logging.Logger`

**Descripción:** Devuelve un logger hijo bajo el espacio de nombres `cicids2017.<name>`. Al heredar del logger raíz configurado por `setup_logger`, el logger hijo recibe automáticamente los handlers de consola y fichero sin configuración adicional.

**Parámetros:**
- `name` — identificador del módulo solicitante. La convención del proyecto es pasar `__name__` en la línea de nivel de módulo: `log = get_logger(__name__)`.

**Retorna:** Un `logging.Logger` cuyo nivel efectivo y handlers se heredan del logger raíz `"cicids2017"`.

**Notas:**
- El uso de `__name__` produce nombres completamente cualificados del tipo `cicids2017.models.isolation_forest`, lo que permite distinguir el origen exacto de cada mensaje en el log.
- Si `setup_logger` no ha sido llamado previamente (p. ej., en tests unitarios), el logger hijo no tendrá handlers y los mensajes serán silenciados por el `lastResort` handler de Python (nivel WARNING hacia stderr), lo cual es un comportamiento seguro y no genera excepciones.

## Dependencias Internas

Ninguna. Este módulo sólo utiliza la biblioteca estándar de Python (`logging`, `sys`, `pathlib`).

## Referencias

- Van Rossum, G. et al. (2002). *PEP 282 – A Logging System*. Python Enhancement Proposals. https://peps.python.org/pep-0282/
- Python Software Foundation. (2024). *logging — Logging facility for Python*. Python 3.10 Documentation. https://docs.python.org/3.10/library/logging.html
- Gerhards, R. (2009). *RFC 5424: The Syslog Protocol*. IETF. https://datatracker.ietf.org/doc/html/rfc5424
