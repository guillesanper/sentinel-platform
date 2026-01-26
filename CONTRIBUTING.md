# Guía de Contribución - Sentinel Platform

Esta guía te ayudará a ejecutar todas las verificaciones de lint y CI localmente antes de hacer push a una rama.

## 📋 Pre-requisitos

Asegúrate de tener activado el virtual environment:

```powershell
# En Windows
.\venv\Scripts\Activate.ps1

# En macOS/Linux
source venv/bin/activate
```

Instala las dependencias de desarrollo:

```powershell
pip install -r requirements-dev.txt
```

---

## ✅ Checklist antes de hacer Push

Ejecuta estas verificaciones en orden antes de hacer push a cualquier rama:

### 1. **Formateo de código** 🎨
Ruff formatea automáticamente el código según el estilo del proyecto:

```powershell
# Formatear automáticamente
ruff format .

# Verificar que esté bien formateado (sin cambios)
ruff format --check .
```

### 2. **Linting** 🔍
Verifica que no haya problemas de código:

```powershell
# Ver problemas detectados
ruff check .

# Ver problemas en formato de GitHub Actions
ruff check . --output-format=github
```

**Problemas comunes:**
- Imports sin usar: Ruff los eliminará automáticamente
- Líneas demasiado largas (>100 caracteres)
- Variables no utilizadas
- Código duplicado

### 3. **Type Checking** 🔐
Verifica los tipos de datos (opcional pero recomendado):

```powershell
mypy api/ ml/ --ignore-missing-imports
```

### 4. **Tests Unitarios** ✔️
Ejecuta todos los tests para asegurar que nada se rompe:

```powershell
# Ejecutar solo tests unitarios
pytest tests/unit/ -v

# Con cobertura de código
pytest tests/unit/ --cov=api --cov=ml --cov-report=html

# Ver el reporte en htmlcov/index.html
```

### 5. **Security Checks** 🔒
Verifica vulnerabilidades de seguridad:

```powershell
# Verificar vulnerabilidades en dependencias
pip-audit

# Análisis de código inseguro
bandit -r api/ ml/ -ll
```

---

## 🚀 Ejecución Rápida (Script)

Si quieres ejecutar todo de una vez, usa este comando:

```powershell
# Formateo + Lint + Type Check + Tests
ruff format . && ruff check . && mypy api/ ml/ --ignore-missing-imports && pytest tests/unit/ -v
```

O ejecuta solo las verificaciones críticas:

```powershell
# Más rápido: formato + lint + tests
ruff format . && ruff check . && pytest tests/unit/ --no-cov -q
```

---

## 📝 Orden Recomendado

1. **Hacer cambios** en el código
2. **Ejecutar `ruff format .`** para formatear automáticamente
3. **Ejecutar `ruff check .`** para ver problemas
4. **Ejecutar `pytest tests/unit/`** para verificar tests
5. **Hacer commit y push** si todo está OK

---

## 🔧 Configuración

Las herramientas se configuran automáticamente desde `pyproject.toml`:

### Ruff
- **Target:** Python 3.11+
- **Línea máxima:** 100 caracteres
- **Reglas activas:** E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, ERA, PL, RUF

### MyPy
- **Modo:** Check basic types
- **Reportes:** Línea y columna

### Pytest
- **Carpeta de tests:** `tests/`
- **Cobertura mínima:** 70%

---

## ⚠️ Errores Comunes y Soluciones

### Error: "ruff: command not found"
```powershell
# Usa python -m en su lugar
python -m ruff check .
python -m ruff format .
```

### Error: "ModuleNotFoundError"
```powershell
# Reinstala las dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Los tests fallan pero el código se ve bien
```powershell
# Ejecuta con modo verbose para ver detalles
pytest tests/unit/ -vv -s
```

### MyPy reporta errores de tipo pero no quiero fallar
```powershell
# MyPy está configurado como opcional en CI
# Continúa adelante, pero es recomendable arreglarlo
```

---

## 🔄 Pipeline CI en GitHub

Cuando hagas push, GitHub ejecutará automáticamente:

1. **Lint check** - `ruff check .`
2. **Format check** - `ruff format --check .`
3. **Type check** - `mypy api/ ml/`
4. **Unit tests** - `pytest tests/unit/`
5. **Coverage** - Debe estar >70%
6. **Security scan** - `pip-audit`, `bandit`
7. **Build Docker** (solo en main)

El pipeline está definido en [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)

---

## 📚 Recursos Útiles

- [Documentación de Ruff](https://docs.astral.sh/ruff/)
- [Documentación de MyPy](https://mypy.readthedocs.io/)
- [Documentación de Pytest](https://docs.pytest.org/)
- [Sentinel Platform README](README.md)

---

## 💡 Tips

- **Antes de hacer PR:** Ejecuta todas las verificaciones
- **Usa pre-commit hooks:** Puedes configurar pre-commit para ejecutar verificaciones automáticamente
- **IDE Integration:** Muchos IDEs (VS Code, PyCharm) integran Ruff y MyPy
- **Commits pequeños:** Haz commits específicos y pequeños para facilitar reviews

---

¿Preguntas? Consulta la documentación del proyecto en `docs/` o crea un issue en GitHub.
