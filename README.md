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

## Data and Methodology

*   **Product Datasets**: Shade data for foundations and concealers are sourced from curated cosmetic datasets, including precise hex codes and brand tiers.
*   **Facial Mapping**: Powered by the MediaPipe Face Landmarker V2 model for sub-millimeter landmark precision.
*   **Wardrobe Classification**: Utilizes zero-shot classification to categorize items without the need for specific training on new clothing styles.

---

## Design Philosophy

Slayr is built on the **Aura-Prism UI** system, a bespoke design framework characterized by glassmorphism, dynamic gradients, and fluid micro-animations. The interface is designed to be data-centric yet visually premium, ensuring that complex analytical results are presented in an accessible and sophisticated manner.

---
**Copyright 2026 Slayr AI Team. All rights reserved.**
