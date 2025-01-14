# **HyperKGE**

## **Introduction**

**HyperKGE** is a project designed to analyze and predict close approaches between satellites by constructing a knowledge graph based on **orbital threat tables**. The knowledge graph includes **satellite nodes** and **orbit nodes**, enabling efficient modeling of satellite behavior and orbit relationships. Additionally, an adjacency matrix for orbits is constructed to facilitate the analysis.



---

## **Requirements**

Ensure the following dependencies are installed before running the project:

- Python >= 3.8
- PyTorch >= 1.8.1
- NumPy
- SciPy
- Pandas
- Networkx

Install the dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## **Data Processing**

Prepare the data by processing the orbital threat table and generating the adjacency matrix. Use the following command:

```bash
python data_process.py
```

---

## **Running Experiments**

Execute the main training script with the following command:

```bash
python -u main.py \
    --dataset SO \
    --ssl_temp 0.5 \
    --ssl_ureg 0.06 \
    --ssl_ireg 0.06 \
    --lr 0.05 \
    --rank 3 \
    --patience 20 \
    --batch 4096 > train.log 2>&1
```

### Explanation of Key Parameters:
- `--dataset`: Dataset name (e.g., `SO` for satellite and orbit data).
- `--ssl_temp`: Temperature parameter for SSL.
- `--ssl_ureg`: Unsupervised regularization strength.
- `--ssl_ireg`: Interaction regularization strength.
- `--lr`: Learning rate.
- `--rank`: Rank of embedding factors.
- `--patience`: Early stopping patience.
- `--batch`: Batch size for training.

---

## **Output**

The training logs are saved to `train.log`. Evaluation results and model checkpoints will be saved in the designated output directory.

---

## **Future Work**

- Extend the knowledge graph to include additional node types such as ground stations.
- Experiment with different hypergraph models and SSL techniques.
- Integrate real-time orbital data for dynamic analysis.

---

## **Contributors**

- Huangchen
- yufei
- ZhejiangLab

---

Let me know if you'd like to refine or add additional sections!