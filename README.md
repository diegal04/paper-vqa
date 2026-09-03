# paper-vqa

Repositorio experimental para estudiar VQA selectivo en imágenes tomadas por personas ciegas. El método combina BLIP-VQA con LoRA y una cabeza auxiliar que estima si una pregunta puede responderse; cuando no supera un umbral calibrado, el sistema se abstiene de generar una respuesta.

El código de esta carpeta es independiente del TFG original. `mi-tfg` no se importa ni se modifica.

## Protocolo científico

- VizWiz `train`: entrenamiento.
- VizWiz `validation`: selección de checkpoint, *sweeps* y calibración del umbral.
- VizWiz `test`: evaluación final una vez congelada la configuración.
- Las fuentes de replay solo pueden usar *splits* disjuntos de cualquier evaluación. Cada ejecución escribe manifiestos de IDs y aborta si detecta solapamientos.

Las respuestas de referencia se conservan completas; la métrica VQA usa el protocolo *leave-one-annotator-out*. Para respondibilidad, la métrica principal es AP. El informe también incluye AUROC, F1, calibración y métricas de VQA selectivo (cobertura, riesgo y tasa insegura de respuesta).

## Instalación con uv

```bash
uv sync --extra dev
```

Descarga las anotaciones e imágenes oficiales de VizWiz y configura sus rutas en `configs/data/vizwiz.yaml`. Los valores de `path`, `image_root`, revisión, muestras máximas, pesos, *splits*, longitudes, optimización y métricas viven en YAML: no hay tamaños de dataset o hiperparámetros escondidos en Python.

## Ejecución

Una comprobación de datos y manifiestos:

```bash
uv run paper-vqa-prepare-data
```

Una ejecución de desarrollo con una semilla:

```bash
uv run paper-vqa-train
```

Ablación sin cabeza, sin cambiar código:

```bash
uv run paper-vqa-train head=disabled
```

Replay configurable y tamaño limitado:

```bash
uv run paper-vqa-train replay=both replay.sources.0.max_samples=3000 replay.sources.1.max_samples=2200
```

Resultados finales con cinco semillas:

```bash
uv run paper-vqa-train -m experiment=final trainer.seed=41,42,43,44,45
```

La evaluación carga un checkpoint ya seleccionado, calibra el umbral en validación y solo entonces evalúa test:

```bash
uv run paper-vqa-evaluate experiment=final evaluation.checkpoint_path=outputs/.../checkpoint
```

Para W&B, cambia `logging.enabled=true` y define `logging.mode=online` (o usa `offline`). Hydra permite que un agente de W&B inyecte los mismos overrides YAML en un *sweep*; los artefactos incluyen configuración resuelta, pesos seguros, manifiestos y predicciones.

## Ablaciones previstas

- BLIP base, LoRA VQA sin cabeza y LoRA + cabeza multitarea.
- `loss=multitask` frente a `loss=answerable_only`.
- Cabeza con `masked_mean` frente a `cls`.
- `replay=none`, `textvqa`, `vqav2` y `both`, con proporciones/pesos y muestras explorados por *sweep*.

Una sola semilla es únicamente para depuración. Los resultados publicables se agregan sobre cinco semillas después de congelar el protocolo de validación.

## Calidad

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

VizWiz se distribuye bajo CC BY 4.0; consulta la documentación oficial para la descarga, licencia y el formato de anotaciones.
