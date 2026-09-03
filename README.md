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

El proyecto fija PyTorch 2.6.0 y Torchvision 0.21.0 desde el índice oficial
CUDA 12.4. Es compatible con el driver NVIDIA local que expone CUDA 12.4; no
uses la rueda CUDA 13 que instalaría PyPI por defecto.

Train y validation se cargan desde las copias de Hugging Face de Multimodal-Fatima. Para test, descarga una vez las anotaciones oficiales recién publicadas:

```bash
curl --fail --location --create-dirs \
  --output data/vizwiz/VQA_test.json \
  https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json
```

El adaptador híbrido une ese JSON por `filename` a las imágenes/preguntas de `Multimodal-Fatima/VizWiz_test`. Los valores de fuentes, revisiones, muestras máximas, pesos, *splits*, longitudes, optimización y métricas viven en YAML: no hay tamaños de dataset o hiperparámetros escondidos en Python.

## Ejecución

Una comprobación de datos y manifiestos:

```bash
uv run paper-vqa-prepare-data
```

Una auditoría descriptiva reproducible de los tres *splits* (respondibilidad,
referencias, longitudes de texto y una muestra de imágenes decodificadas):

```bash
uv run paper-vqa-audit-data
```

El informe `data_audit.json` y los manifiestos quedan en el directorio de salida
de Hydra. Para no abrir imágenes de test durante una auditoría de desarrollo,
usa `audit.include_test=false`.

Por defecto esto solo valida train y validation. La comprobación explícita de test —que descarga sus imágenes desde Hugging Face— es:

```bash
uv run paper-vqa-prepare-data data.include_test=true
```

Una ejecución de desarrollo con una semilla:

```bash
uv run paper-vqa-train
```

El baseline preentrenado, sin LoRA ni cabeza y exclusivamente en validation,
se ejecuta primero como prueba corta y después completo:

```bash
uv run paper-vqa-baseline model=blip_vqa_zero_shot head=disabled \
  experiment=baseline_smoke baseline.max_samples=20
uv run paper-vqa-baseline model=blip_vqa_zero_shot head=disabled experiment=baseline
```

El comando rechaza explícitamente LoRA, la cabeza auxiliar y el split test.

Para medir VQA Accuracy de un checkpoint durante desarrollo sin abrir test:

```bash
uv run paper-vqa-evaluate-development head=disabled \
  evaluation.checkpoint_path=outputs/.../checkpoint \
  experiment.name=lora_vqa_development
```

Este comando está restringido a `data.validation`; no calibra umbrales de una
cabeza sin que se indique `development.threshold` de forma explícita.

Los comandos de entrenamiento y evaluación muestran barras de progreso con
porcentaje, velocidad, ETA y pérdida media. Puedes ocultarlas con
`trainer.progress.enabled=false` o mantener las barras completadas con
`trainer.progress.leave=true`.

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
