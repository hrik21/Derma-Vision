# Design — DermaVision Architecture (Code Form)

This file contains the **diagram in code form** (Mermaid) and supporting design notes so you can render the architecture diagram directly on GitHub (or any Markdown viewer that supports Mermaid).

> **How to use**
> - GitHub now supports Mermaid diagrams in markdown. Just add this `DESIGN.md` to your repo and view it on GitHub.
> - If your viewer does not render Mermaid, you can paste the Mermaid code into an online Mermaid live editor (https://mermaid.live) to visualize and export as PNG/SVG.

---

## Mermaid Diagram (Flowchart)

```mermaid
flowchart TD
    %% Styling
    classDef box fill:#ffffff,stroke:#cfd8dc,stroke-width:2px,rx:8,ry:8;
    classDef stage fill:#f1f8ff,stroke:#90caf9,stroke-width:1px,rx:6,ry:6;
    classDef modelbox fill:#fff8e1,stroke:#ffb74d,stroke-width:1px,rx:6,ry:6;
    classDef metricbox fill:#fff3e0,stroke:#ffcc80,stroke-width:1px,rx:6,ry:6;
    classDef db fill:#e8f5e9,stroke:#66bb6a,stroke-width:1px,rx:6,ry:6;

    %% Nodes
    A[📁 HAM10000 Dataset<br/>(10,015 images)]:::db
    B[⚙️ Preprocessing<br/>Resize 176x176, Normalize, Augment]:::stage
    subgraph TRAIN["🧪 Model Training"]
      direction LR
      M1[Baseline CNN<br/>(2 conv layers)]:::modelbox
      M2[Custom CNN<br/>(4 conv layers)]:::modelbox
      M3[ResNet50 (Transfer Learning)]:::modelbox
    end
    C[📊 Evaluation<br/>Accuracy, Precision, Recall, F1, Confusion Matrix]:::metricbox
    D[🔮 Prediction & Visualization<br/>Reports, Grad-CAM, Dashboards]:::stage

    %% Flows
    A --> B --> TRAIN --> C --> D

    %% Additional notes
    subgraph NOTES["Design Notes"]
      N1["• Input size: 176x176x3"] 
      N2["• Optimizer: Adam<br/>• Loss: Categorical Cross-Entropy"]
      N3["• Train/Val/Test split: 5187 / 1556 / 667"]
      N4["• Best model observed: Custom CNN (Test acc 67.76%)"]
    end
    NOTES -.-> A
    NOTES -.-> M2

    style TRAIN stroke:#90a4ae,stroke-width:2px,fill:#ffffff
```

---

## SVG alternative (Mermaid -> SVG)
If you prefer an **SVG** version, paste the Mermaid code above into [Mermaid Live Editor](https://mermaid.live), then export as SVG or PNG.

---

## Design Rationale (short)
- **Three-model comparison**: Allows exploration of simple vs deeper custom architectures vs transfer learning.  
- **Preprocessing emphasis**: Resizing, normalization, and augmentation are critical to reduce overfitting and improve generalization on the HAM10000 dataset.  
- **Evaluation stage**: Use per-class metrics and confusion matrices due to class imbalance. Report test-set results for model selection.  
- **Visualization & interpretability**: Add Grad-CAM to highlight lesion regions that drive predictions and provide interpretability for clinicians.

---

## File placement suggestion
- Add `DESIGN.md` to the repo root or `/docs` folder.
- Keep the architecture PNG (if you want the visual image) in `/images/dermavision_architecture.png` and reference it from `README.md`.

---

## Ready-to-copy snippet
Below is the Mermaid snippet alone if you only want the diagram:

\`\`\`mermaid
flowchart TD
    classDef box fill:#ffffff,stroke:#cfd8dc,stroke-width:2px,rx:8,ry:8;
    classDef stage fill:#f1f8ff,stroke:#90caf9,stroke-width:1px,rx:6,ry:6;
    classDef modelbox fill:#fff8e1,stroke:#ffb74d,stroke-width:1px,rx:6,ry:6;
    classDef metricbox fill:#fff3e0,stroke:#ffcc80,stroke-width:1px,rx:6,ry:6;
    classDef db fill:#e8f5e9,stroke:#66bb6a,stroke-width:1px,rx:6,ry:6;

    A[📁 HAM10000 Dataset<br/>(10,015 images)]:::db
    B[⚙️ Preprocessing<br/>Resize 176x176, Normalize, Augment]:::stage
    subgraph TRAIN["🧪 Model Training"]
      direction LR
      M1[Baseline CNN<br/>(2 conv layers)]:::modelbox
      M2[Custom CNN<br/>(4 conv layers)]:::modelbox
      M3[ResNet50 (Transfer Learning)]:::modelbox
    end
    C[📊 Evaluation<br/>Accuracy, Precision, Recall, F1, Confusion Matrix]:::metricbox
    D[🔮 Prediction & Visualization<br/>Reports, Grad-CAM, Dashboards]:::stage

    A --> B --> TRAIN --> C --> D
\`\`\`

---

*If you'd like, I can also commit this `DESIGN.md` directly into your repo (I can show the `git` commands), or generate a ready-to-upload SVG/PNG exported from Mermaid for you.*  
