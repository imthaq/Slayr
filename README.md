# Slayr | Advanced AI Aesthetic Engine

**Live Demo**: [slayr.onrender.com](https://slayr.onrender.com/)

Slayr is a premium aesthetic consultancy platform that utilizes artificial intelligence and color science to curate personalized style blueprints. By integrating advanced facial landmarking, zero-shot image classification, and colorimetric analysis, Slayr provides a comprehensive suite of tools for individuals seeking to refine their visual identity.

---

## Core Features

### Smart Match Precision
The Smart Match engine utilizes the industry-standard CIE2000 Delta E algorithm to provide ultra-precise cosmetic shade matching. By analyzing skin tone frequencies within the CIELAB color space, the system identifies the most compatible foundation and concealer shades across an extensive database of premium and drugstore brands.

### Chroma Skin and Vibe Check
This module performs objective undertone detection (Cool, Warm, Neutral) and assigns users to one of the twelve Seasonal Color Theory archetypes. Using LAB space analysis, it generates scientifically grounded clothing palettes that harmonize with the user's natural chromatic signature.

### Morpho Face Architecture
Morpho Face employs MediaPipe facial landmarking to perform geometric mapping of the user's facial structure. It accurately classifies face shapes—such as Oval, Square, Round, Heart, Diamond, and Oblong—providing the foundation for proportional eyewear and grooming recommendations.

### Frame Fit AR Synthesis
Frame Fit offers an augmented reality experience for eyewear selection. By calculating eye spacing and facial width, the system overlays structurally complementary frames onto the user's portrait, allowing for objective evaluation of different styles.

### Grooming Blueprint
The Grooming module provides face-shape-optimized recommendations for beard and hair styles. Using facial geometry, it identifies styles that create structural balance and enhance the user's natural bone structure.

### Capsule Wardrobe Engine
The Wardrobe system leverages the CLIP (Contrastive Language-Image Pre-training) model for zero-shot classification of uploaded clothing items. It extracts color palettes, calculates harmony scores using Delta E, and utilizes a multi-pass diversity engine to generate unique, neural-curated outfit combinations.

---

## Technical Architecture

*   **Backend**: Python / Flask
*   **Frontend**: HTML5, Vanilla JavaScript, Aura-Prism CSS System
*   **Database**: SQLite with Flask-SQLAlchemy
*   **Computer Vision**: OpenCV, MediaPipe Face Landmarker V2
*   **Machine Learning**: CLIP (via Transformers), K-Means Clustering
*   **Color Science**: CIELAB Delta E (CIE2000), Google Monk Skin Tone (MST) Scale
*   **Progressive Web App**: Service Worker integration for offline capabilities and desktop/mobile installation.

---

## Installation and Deployment

1.  **Clone the Repository**:
    ```bash
    git clone [repository-url]
    cd slayr_final
    ```

2.  **Environment Setup**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # source .venv/bin/activate # macOS/Linux
    ```

3.  **Dependency Installation**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Database Initialization**:
    The system automatically synchronizes product data upon the first launch. To manually trigger a sync:
    ```bash
    python sync_shades.py
    ```

5.  **Execution**:
    ```bash
    python app.py
    ```
    The platform will be available at `http://127.0.0.1:5000`.

---

## Running with Docker

The project ships with a `Dockerfile` that installs all system and Python dependencies (including CPU-only PyTorch, MediaPipe, and OpenCV) and prefetches the CLIP model weights at build time.

1.  **Build the image**:
    ```bash
    docker build -t slayr .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 5000:5000 slayr
    ```
    The platform will be available at `http://127.0.0.1:5000`.

3.  **Run with a persistent database** (optional — keeps your SQLite data between container restarts instead of re-syncing from CSV every time):
    ```bash
    docker run -p 5000:5000 -v slayr_instance:/app/instance slayr
    ```

> **Note on memory**: Docker Desktop (and most local machines) give containers several GB of RAM by default, which is enough for the full stack — including CLIP-based Wardrobe classification — to run correctly. This differs from the deployed Render instance; see the note below.

---

## Deployment Notes (Render Free Tier)

The live demo runs on Render's free tier, which caps memory at 512MB. The full ML stack (PyTorch, Transformers/CLIP, MediaPipe, OpenCV) run together is tight against that ceiling, so **on Render specifically, automatic AI classification in the Capsule Wardrobe Engine may fail and return "Unknown" for uploaded items** — this is a memory constraint of the free hosting tier, not a bug in the classification logic itself (confirmed working correctly when run locally or in Docker with normal RAM).

**Workaround**: every wardrobe item has a category dropdown next to it — if an item lands in "Unknown," simply select the correct category (Top, Bottom, Outerwear, Shoes, Accessory, Full) manually from the dropdown. The rest of the wardrobe engine (palette extraction, color harmony scoring, outfit generation) is unaffected and works normally regardless of how the category was set.

---

## Data and Methodology

*   **Product Datasets**: Shade data for foundations and concealers are sourced from curated cosmetic datasets, including precise hex codes and brand tiers.
*   **Facial Mapping**: Powered by the MediaPipe Face Landmarker V2 model for sub-millimeter landmark precision.
*   **Wardrobe Classification**: Utilizes zero-shot classification to categorize items without the need for specific training on new clothing styles.

---

## Design Philosophy

Slayr is built on the **Aura-Prism UI** system, a bespoke design framework characterized by glassmorphism, dynamic gradients, and fluid micro-animations. The interface is designed to be data-centric yet visually premium, ensuring that complex analytical results are presented in an accessible and sophisticated manner.

## Contributors
*   [@imthaq](https://github.com/imthaq)
*   [@Laibabasharat-26](https://github.com/Laibabasharat-26)
*   [@Anamta-Tariq](https://github.com/Anamta-Tariq)

---
**Copyright 2026 Slayr AI Team. All rights reserved.**
