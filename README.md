# Estructura de Quadtree en Renderizado Adaptativo de Teselación

Este documento resume la estructura de datos basada en quadtree y la lógica relacionada descrita en el artículo *Improved Adaptive Tessellation Rendering Algorithm* de Wang et al. (2023). Sirve como referencia para implementar un MVP simplificado del componente quadtree.

## Visión General

El artículo describe un algoritmo de **teselación adaptativa** con dos etapas principales:

- **Fase offline**: Lee datos de malla y construye una jerarquía de subdivisión usando un quadtree.
- **Fase online**: Utiliza el quadtree para evaluar parches de superficie eficientemente en tiempo real usando shaders.

El **quadtree** almacena información sobre la subdivisión de parches de superficie y hace referencia a plantillas de puntos de control.

## Tipos de Nodos del Quadtree

Cada nodo en el quadtree representa una región de una cara de malla. Los nodos hoja representan subdominios que pueden evaluarse.

- `RegularNode`: Parche normal con 16 puntos de control.
- `CreaseNode`: Parche con pliegues semi-agudos, almacena nitidez del pliegue.
- `SpecialNode`: Para esquinas con topología no convencional (vértice extremo + tangentes).
- `TerminalNode`: Detiene la subdivisión; almacena 24 puntos de control en una malla 5x5 y un índice de rotación.

## Estructura de Nodos

### Nodo Interno
- Contiene 4 hijos (estructura estándar de quadtree).
- Cada nivel representa un paso de subdivisión recursiva.

### Nodo Hoja (Uno de los siguientes)
- `RegularNode`: Contiene 16 puntos de control.
- `CreaseNode`: Similar al regular pero incluye metadatos de nitidez de pliegue.
- `SpecialNode`: 3 plantillas para evaluar un parche de esquina.
- `TerminalNode`: Malla densa de control, sin subdivisión adicional.

## Plantillas y Puntos de Control

- Las plantillas codifican puntos de control como **sumas ponderadas de vértices del anillo vecino**.
- Cada cara mantiene una lista ordenada de vértices del anillo vecino.
- Una **matriz de pesos** mapea índices de vértices a pesos de puntos de control.
- Esta estructura permite reutilización y eficiencia durante la evaluación en tiempo de ejecución.

## Flujo de Evaluación (Simplificado)

1. **Offline**
   - Construir el quadtree basado en subdivisión Catmull-Clark.
   - Adjuntar plantillas y referencias de pesos a los nodos.

2. **Online**
   - Recorrer el quadtree para encontrar el nodo de evaluación basado en coordenadas UV.
   - Evaluar el parche de superficie usando plantillas y pesos de puntos de control.
   - Para nodos terminales/especiales, usar lógica fija para interpolación de tangentes/posiciones.

## Algoritmos Mencionados

- **Recorrido greedy** a través del quadtree para optimizar divergencia SIMT.
- **Matriz de adyacencia de pesos** para suma ordenada sensible de punto flotante.
- **Montículo de Fibonacci** (conceptual) para optimización teórica del tiempo de recorrido (opcional para MVP).

## Plan de Implementación Sugerido

1. Definir la clase `QuadtreeNode` y sus tipos (enum o subclases).
2. Definir la estructura `Template` (lista de pesos + índices de vértices).
3. Construir una malla simple y subdividirla recursivamente para formar el árbol.
4. Implementar métodos `insert`, `subdivide`, `query`.
5. Escribir pruebas unitarias para:
   - Generación correcta de plantillas.
   - Profundidad de subdivisión.
   - Precisión de consulta en bordes, centros y esquinas.

## Referencias

Wang, M. et al. (2023). *Improved Adaptive Tessellation Rendering Algorithm*. Technology and Health Care, 31(S81–S95). DOI: 10.3233/THC-236009
