# MT-VQA: Multi-Table Visual Question Answering

This repository provides the pipeline to process raw table datasets, build structural maps, generate graph indices, and evaluate models.

---

## 🚀 Getting Started

### 1. Environment Setup
First, set up your environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Usage

### 2. Build Structural Maps from Raw Datasets
Generate structural maps from the raw table images using the structural index builder.

* **For ComTQA dataset:**
  ```bash
  python -c "
  from graph_src.indexing.table_structural_index import TableStructuralIndexBuilder
  builder = TableStructuralIndexBuilder(device='cuda:0')
  builder.index_directory(
      image_dir='data/raw/comtqa/images',
      output_dir='data/indices/structural_maps_comtqa',
      skip_existing=False,
  )
  "
  ```

* **For TableVQA-Bench dataset:**
  ```bash
  python -c "
  from graph_src.indexing.table_structural_index import TableStructuralIndexBuilder
  builder = TableStructuralIndexBuilder(device='cuda:0')
  builder.index_directory(
      image_dir='data/raw/tablevqa_bench/images/fintabnetqa',
      output_dir='data/indices/structural_maps_tablevqa_bench/fintabnetqa',
      skip_existing=False,
  )
  "
  ```

---

### 3. Build Graphs from Structural Maps
Construct table graphs using the generated structural map directories.

* **For ComTQA dataset:**
  ```bash
  python -m graph_src.indexing.graph_indexer \
      --maps_dir data/indices/structural_maps_comtqa \
      --output data/indices/table_graph_comtqa.gpickle \
      --min_weight 0.08
  ```

* **For TableVQA-Bench dataset:**
  ```bash
  python -m graph_src.indexing.graph_indexer \
      --maps_dir data/indices/structural_maps_tablevqa_bench \
      --output data/indices/table_graph_tablevqa_bench.gpickle \
      --min_weight 0.08
  ```

---

### 4. Answer Generation and Evaluation
Run inference using the generated structural maps and table graphs to evaluate execution performance.

* **For ComTQA dataset:**
  ```bash
  python -m graph_src.evaluation.run_comtqa_graph \
      --dataset_dir data/raw/comtqa \
      --maps_dir data/indices/structural_maps_comtqa \
      --graph_path data/indices/table_graph_comtqa.gpickle \
      --device cuda:0 \
      --output_dir results/graph_comtqa
  ```

* **For TableVQA-Bench dataset:**
  ```bash
  python -m graph_src.evaluation.run_tablevqa_graph \
      --dataset_dir data/raw/tablevqa_bench \
      --maps_dir data/indices/structural_maps_tablevqa_bench \
      --graph_path data/indices/table_graph_tablevqa_bench.gpickle \
      --device cuda:0 \
      --output_dir results/graph_tablevqa_bench
  ```
