MST_LAB = [
    (94.2, -0.6, 9.3, "MST 1"),
    (92.3, -1.0, 11.2, "MST 2"),
    (93.1, 0.2, 14.2, "MST 3"),
    (87.6, 0.5, 17.7, "MST 4"),
    (77.9, 3.5, 23.1, "MST 5"),
    (55.1, 7.8, 26.7, "MST 6"),
    (42.5, 12.3, 20.5, "MST 7"),
    (30.7, 11.7, 13.3, "MST 8"),
    (21.1, 2.7, 6.0, "MST 9"),
    (14.6, 1.5, 3.5, "MST 10"),
]

SEASON_LAB = [
    # ── Light Spring (20 pts) ── L: 78-92, a: 2-12, b: 10-22 ──
    (88, 4, 15, "Light Spring"), (82, 6, 18, "Light Spring"), (75, 8, 20, "Light Spring"),
    (92, 2, 10, "Light Spring"), (91, 3, 12, "Light Spring"), (90, 3, 14, "Light Spring"),
    (89, 4, 11, "Light Spring"), (87, 5, 16, "Light Spring"), (86, 5, 13, "Light Spring"),
    (85, 6, 17, "Light Spring"), (84, 7, 19, "Light Spring"), (83, 7, 15, "Light Spring"),
    (81, 8, 21, "Light Spring"), (80, 9, 20, "Light Spring"), (79, 10, 22, "Light Spring"),
    (78, 11, 18, "Light Spring"), (90, 5, 19, "Light Spring"), (85, 4, 12, "Light Spring"),
    (83, 9, 14, "Light Spring"), (80, 12, 16, "Light Spring"),

    # ── True Spring (20 pts) ── L: 62-82, a: 8-20, b: 20-36 ──
    (78, 12, 28, "True Spring"), (72, 15, 32, "True Spring"), (68, 18, 35, "True Spring"),
    (82, 8, 20, "True Spring"), (80, 9, 22, "True Spring"), (79, 10, 24, "True Spring"),
    (77, 11, 26, "True Spring"), (76, 12, 30, "True Spring"), (75, 13, 25, "True Spring"),
    (74, 14, 34, "True Spring"), (73, 14, 28, "True Spring"), (71, 16, 36, "True Spring"),
    (70, 16, 30, "True Spring"), (69, 17, 33, "True Spring"), (67, 18, 27, "True Spring"),
    (66, 19, 31, "True Spring"), (65, 19, 23, "True Spring"), (64, 20, 29, "True Spring"),
    (63, 20, 35, "True Spring"), (62, 15, 21, "True Spring"),

    # ── Bright Spring (20 pts) ── L: 65-82, a: 15-28, b: 18-30 ──
    (80, 20, 25, "Bright Spring"), (75, 22, 22, "Bright Spring"), (70, 25, 28, "Bright Spring"),
    (82, 15, 18, "Bright Spring"), (81, 16, 20, "Bright Spring"), (79, 17, 23, "Bright Spring"),
    (78, 18, 19, "Bright Spring"), (77, 19, 26, "Bright Spring"), (76, 20, 21, "Bright Spring"),
    (74, 21, 24, "Bright Spring"), (73, 22, 27, "Bright Spring"), (72, 23, 20, "Bright Spring"),
    (71, 24, 30, "Bright Spring"), (69, 25, 22, "Bright Spring"), (68, 26, 26, "Bright Spring"),
    (67, 27, 29, "Bright Spring"), (66, 28, 24, "Bright Spring"), (65, 18, 19, "Bright Spring"),
    (76, 16, 28, "Bright Spring"), (72, 27, 21, "Bright Spring"),

    # ── Light Summer (20 pts) ── L: 75-92, a: 5-18, b: 2-14 ──
    (88, 8, 6, "Light Summer"), (82, 12, 10, "Light Summer"), (78, 15, 12, "Light Summer"),
    (92, 5, 2, "Light Summer"), (91, 6, 4, "Light Summer"), (90, 6, 3, "Light Summer"),
    (89, 7, 5, "Light Summer"), (87, 8, 7, "Light Summer"), (86, 9, 8, "Light Summer"),
    (85, 10, 6, "Light Summer"), (84, 10, 9, "Light Summer"), (83, 11, 11, "Light Summer"),
    (81, 13, 10, "Light Summer"), (80, 14, 13, "Light Summer"), (79, 14, 8, "Light Summer"),
    (77, 16, 14, "Light Summer"), (76, 17, 12, "Light Summer"), (75, 18, 11, "Light Summer"),
    (90, 8, 9, "Light Summer"), (85, 13, 5, "Light Summer"),

    # ── True Summer (20 pts) ── L: 60-78, a: 8-18, b: -2-10 ──
    (75, 10, 5, "True Summer"), (70, 12, 4, "True Summer"), (65, 15, 6, "True Summer"),
    (78, 8, 0, "True Summer"), (77, 9, 2, "True Summer"), (76, 9, -1, "True Summer"),
    (74, 10, 3, "True Summer"), (73, 11, 7, "True Summer"), (72, 11, 1, "True Summer"),
    (71, 12, 8, "True Summer"), (69, 13, 6, "True Summer"), (68, 13, -2, "True Summer"),
    (67, 14, 3, "True Summer"), (66, 14, 9, "True Summer"), (64, 15, 0, "True Summer"),
    (63, 16, 5, "True Summer"), (62, 17, 10, "True Summer"), (61, 17, 2, "True Summer"),
    (60, 18, 7, "True Summer"), (74, 16, -1, "True Summer"),

    # ── Soft Summer (20 pts) ── L: 55-75, a: 10-22, b: 8-20 ──
    (72, 14, 14, "Soft Summer"), (68, 16, 16, "Soft Summer"), (62, 18, 18, "Soft Summer"),
    (75, 10, 8, "Soft Summer"), (74, 11, 10, "Soft Summer"), (73, 12, 12, "Soft Summer"),
    (71, 13, 9, "Soft Summer"), (70, 14, 15, "Soft Summer"), (69, 15, 11, "Soft Summer"),
    (67, 15, 13, "Soft Summer"), (66, 16, 17, "Soft Summer"), (65, 17, 10, "Soft Summer"),
    (64, 18, 19, "Soft Summer"), (63, 19, 15, "Soft Summer"), (61, 19, 12, "Soft Summer"),
    (60, 20, 20, "Soft Summer"), (58, 20, 14, "Soft Summer"), (57, 21, 16, "Soft Summer"),
    (56, 22, 18, "Soft Summer"), (55, 22, 9, "Soft Summer"),

    # ── Soft Autumn (20 pts) ── L: 50-70, a: 6-18, b: 18-30 ──
    (65, 10, 25, "Soft Autumn"), (60, 12, 28, "Soft Autumn"), (55, 14, 22, "Soft Autumn"),
    (70, 6, 18, "Soft Autumn"), (69, 7, 20, "Soft Autumn"), (68, 8, 22, "Soft Autumn"),
    (67, 9, 19, "Soft Autumn"), (66, 9, 24, "Soft Autumn"), (64, 10, 27, "Soft Autumn"),
    (63, 11, 21, "Soft Autumn"), (62, 11, 26, "Soft Autumn"), (61, 13, 30, "Soft Autumn"),
    (59, 13, 23, "Soft Autumn"), (58, 14, 29, "Soft Autumn"), (57, 15, 20, "Soft Autumn"),
    (56, 16, 26, "Soft Autumn"), (54, 17, 24, "Soft Autumn"), (53, 17, 19, "Soft Autumn"),
    (52, 18, 28, "Soft Autumn"), (50, 18, 21, "Soft Autumn"),

    # ── True Autumn (20 pts) ── L: 40-58, a: 14-26, b: 25-40 ──
    (55, 18, 35, "True Autumn"), (50, 20, 38, "True Autumn"), (45, 22, 32, "True Autumn"),
    (58, 14, 25, "True Autumn"), (57, 15, 28, "True Autumn"), (56, 16, 30, "True Autumn"),
    (54, 17, 33, "True Autumn"), (53, 18, 37, "True Autumn"), (52, 18, 27, "True Autumn"),
    (51, 19, 40, "True Autumn"), (49, 20, 34, "True Autumn"), (48, 21, 29, "True Autumn"),
    (47, 21, 36, "True Autumn"), (46, 22, 26, "True Autumn"), (44, 23, 39, "True Autumn"),
    (43, 24, 31, "True Autumn"), (42, 24, 35, "True Autumn"), (41, 25, 28, "True Autumn"),
    (40, 26, 38, "True Autumn"), (40, 14, 26, "True Autumn"),

    # ── Warm Autumn (20 pts) ── L: 32-52, a: 12-24, b: 20-35 ──
    (48, 15, 28, "Warm Autumn"), (42, 18, 32, "Warm Autumn"), (38, 20, 25, "Warm Autumn"),
    (52, 12, 20, "Warm Autumn"), (51, 13, 22, "Warm Autumn"), (50, 13, 26, "Warm Autumn"),
    (49, 14, 24, "Warm Autumn"), (47, 15, 30, "Warm Autumn"), (46, 16, 21, "Warm Autumn"),
    (45, 16, 27, "Warm Autumn"), (44, 17, 35, "Warm Autumn"), (43, 17, 23, "Warm Autumn"),
    (41, 18, 29, "Warm Autumn"), (40, 19, 33, "Warm Autumn"), (39, 19, 22, "Warm Autumn"),
    (37, 20, 31, "Warm Autumn"), (36, 21, 26, "Warm Autumn"), (35, 22, 34, "Warm Autumn"),
    (34, 23, 24, "Warm Autumn"), (32, 24, 30, "Warm Autumn"),

    # ── Deep Winter (20 pts) ── L: 18-40, a: 10-24, b: -5-8 ──
    (35, 15, 2, "Deep Winter"), (30, 18, 0, "Deep Winter"), (25, 20, -2, "Deep Winter"),
    (40, 10, 3, "Deep Winter"), (39, 11, 5, "Deep Winter"), (38, 12, 1, "Deep Winter"),
    (37, 13, 8, "Deep Winter"), (36, 14, -1, "Deep Winter"), (34, 14, 6, "Deep Winter"),
    (33, 15, -3, "Deep Winter"), (32, 16, 4, "Deep Winter"), (31, 17, 7, "Deep Winter"),
    (29, 17, -4, "Deep Winter"), (28, 18, 2, "Deep Winter"), (27, 19, 5, "Deep Winter"),
    (26, 19, -1, "Deep Winter"), (24, 20, 3, "Deep Winter"), (22, 21, -5, "Deep Winter"),
    (20, 22, 1, "Deep Winter"), (18, 24, 0, "Deep Winter"),

    # ── True Winter (20 pts) ── L: 30-50, a: 8-18, b: -10-2 ──
    (45, 10, -5, "True Winter"), (40, 12, -8, "True Winter"), (35, 14, -6, "True Winter"),
    (50, 8, 0, "True Winter"), (49, 9, -2, "True Winter"), (48, 9, 2, "True Winter"),
    (47, 10, -3, "True Winter"), (46, 10, 1, "True Winter"), (44, 11, -7, "True Winter"),
    (43, 11, -1, "True Winter"), (42, 12, -4, "True Winter"), (41, 13, -9, "True Winter"),
    (39, 13, 0, "True Winter"), (38, 14, -10, "True Winter"), (37, 15, -3, "True Winter"),
    (36, 15, 2, "True Winter"), (34, 16, -7, "True Winter"), (33, 17, -1, "True Winter"),
    (32, 17, -9, "True Winter"), (30, 18, -5, "True Winter"),

    # ── Bright Winter (20 pts) ── L: 40-60, a: 18-34, b: 0-12 ──
    (55, 25, 5, "Bright Winter"), (50, 28, 8, "Bright Winter"), (45, 30, 5, "Bright Winter"),
    (60, 18, 1, "Bright Winter"), (59, 19, 4, "Bright Winter"), (58, 20, 0, "Bright Winter"),
    (57, 21, 7, "Bright Winter"), (56, 22, 3, "Bright Winter"), (54, 23, 10, "Bright Winter"),
    (53, 24, 2, "Bright Winter"), (52, 25, 9, "Bright Winter"), (51, 26, 6, "Bright Winter"),
    (49, 27, 11, "Bright Winter"), (48, 28, 3, "Bright Winter"), (47, 29, 12, "Bright Winter"),
    (46, 30, 7, "Bright Winter"), (44, 31, 4, "Bright Winter"), (43, 32, 10, "Bright Winter"),
    (42, 33, 2, "Bright Winter"), (40, 34, 8, "Bright Winter"),
]

def get_mst_label(level_idx):
    if 0 <= level_idx < len(MST_LAB):
        return MST_LAB[level_idx][3]
    return "Unknown"

GROOMING_RECOMMENDATIONS = {
    "Oval": {
        "hair_male": ["Pompadour", "Side Part", "Quiff", "Slick Back"],
        "hair_female": ["LongWaves", "Lob(LongBob)", "BluntBob", "DeepSidePart"],
        "beard": ["Stubble", "Short Beard", "Clean Shaven"],
        "desc": "Balanced proportions allow for maximum versatility. Maintain height on top for a sharp look."
    },
    "Square": {
        "hair_male": ["Buzz Cut", "Undercut", "Short Fringe", "Side Part"],
        "hair_female": ["LongLayers", "SideSweptBangs", "SoftWaves", "TexturedBob"],
        "beard": ["Full Beard", "Short Boxed Beard", "Circle Beard"],
        "desc": "Soften the strong jawline with rounded styles or embrace the structure with sharp, short cuts."
    },
    "Round": {
        "hair_male": ["Pompadour", "Quiff", "Volume on Top", "Faux Hawk"],
        "hair_female": ["LongLayers", "PixieCut", "AsymmetricalCut", "Waves"],
        "beard": ["Van Dyke Beard", "Anchor Beard", "Goatee"],
        "desc": "Create the illusion of length by adding height on top and keeping the sides short."
    },
    "Heart": {
        "hair_male": ["Side-Swept Fringe _ Messy Fringe", "Longer Hair", "Side Part"],
        "hair_female": ["Bob", "SideSweptFringe", "LongWaves", "SoftLayers"],
        "beard": ["Full Beard", "Heavy Stubble", "Scruffy Beard"],
        "desc": "Add width to the narrow chin with a fuller beard and use softer hair styles to balance the forehead."
    },
    "Diamond": {
        "hair_male": ["Faux Hawk", "Quiff", "Side Part", "Longer Hair"],
        "hair_female": ["DeepSidePart", "LongLayers", "TexturedBob", "SidePartedWaves"],
        "beard": ["Full Beard", "Short Beard", "Heavy Stubble"],
        "desc": "Soften the cheekbones and add width to the chin and forehead with textured, layered styles."
    },
    "Oblong": {
        "hair_male": ["Side Part", "Short Fringe", "Side-Swept Fringe _ Messy Fringe"],
        "hair_female": ["BluntBob", "Waves", "SideSweptBangs", "MessyBun"],
        "beard": ["Stubble", "Moustache", "Trimmed Stubble"],
        "desc": "Avoid adding too much height. Choose styles that add width to the sides of the face."
    },
    "Rectangle": {
        "hair_male": ["Side Part", "Short Fringe", "Slick Back"],
        "hair_female": ["BluntBob", "SoftWaves", "SideSweptBangs", "LongLayersWithBangs"],
        "beard": ["Short Beard", "Stubble", "Trimmed Stubble"],
        "desc": "Balance the face length with side-focused volume and avoid extremely high-top styles."
    },
    "Triangle": {
        "hair_male": ["Volume on Top", "Quiff", "Faux Hawk"],
        "hair_female": ["VolumeOnTop", "TexturedPixie", "ShagCut", "LongLayers"],
        "beard": ["Short Boxed Beard", "Trimmed Sides", "Clean Shaven"],
        "desc": "Add volume to the top to balance a wider jawline. Keep the beard neatly trimmed."
    },
    "Pear": {
        "hair_male": ["Volume on Top", "Pompadour", "Quiff"],
        "hair_female": ["VolumeOnTop", "ShagCut", "TexturedPixie", "LongLayers"],
        "beard": ["Van Dyke Beard", "Goatee", "Anchor Beard"],
        "desc": "Focus on volume at the temples and top to create a more balanced silhouette."
    }
}
