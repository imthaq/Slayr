# Monk Skin Tone (MST) Reference Data (CIELAB)
# Source: Google Monk Skin Tone Research
MST_LAB = [
    (94.2, -0.6, 9.3, "MST 1", "#f6ede4"),
    (92.3, -1.0, 11.2, "MST 2", "#f3e7db"),
    (93.1, 0.2, 14.2, "MST 3", "#f7ead0"),
    (87.6, 0.5, 17.7, "MST 4", "#eadaba"),
    (77.9, 3.5, 23.1, "MST 5", "#d7bd96"),
    (55.1, 7.8, 26.7, "MST 6", "#a07e56"),
    (42.5, 12.3, 20.5, "MST 7", "#825c43"),
    (30.7, 11.7, 13.3, "MST 8", "#604134"),
    (21.1, 2.7, 6.0, "MST 9", "#3a312a"),
    (14.6, 1.5, 3.5, "MST 10", "#292420")
]

# Seasonal Reference Points (Standard CIELAB: L 0-100, a -128 to 127, b -128 to 127)
SEASON_LAB = [
    # --- SPRING (Warm, Clear, Light) - Yellow dominant (b > a)
    (88, 4, 15, "Light Spring"), (82, 6, 18, "Light Spring"), (75, 8, 20, "Light Spring"),
    (78, 12, 28, "True Spring"), (72, 15, 32, "True Spring"), (68, 18, 35, "True Spring"),
    (80, 20, 25, "Bright Spring"), (75, 22, 22, "Bright Spring"), (70, 25, 28, "Bright Spring"),
    
    # --- SUMMER (Cool, Soft, Light) - Pink dominant or balanced (a >= b)
    (88, 8, 6, "Light Summer"), (82, 12, 10, "Light Summer"), (78, 15, 12, "Light Summer"),
    (75, 10, 5, "True Summer"), (70, 12, 4, "True Summer"), (65, 15, 6, "True Summer"),
    (72, 14, 14, "Soft Summer"), (68, 16, 16, "Soft Summer"), (62, 18, 18, "Soft Summer"),
    
    # --- AUTUMN (Warm, Soft, Dark) - Heavily yellow/orange (b > a + 5)
    (65, 10, 25, "Soft Autumn"), (60, 12, 28, "Soft Autumn"), (55, 14, 22, "Soft Autumn"),
    (55, 18, 35, "True Autumn"), (50, 20, 38, "True Autumn"), (45, 22, 32, "True Autumn"),
    (48, 15, 28, "Warm Autumn"), (42, 18, 32, "Warm Autumn"), (38, 20, 25, "Warm Autumn"),
    
    # --- WINTER (Cool, Clear, Dark) - High contrast pink/red vs grey (a > b)
    (35, 15, 2, "Deep Winter"), (30, 18, 0, "Deep Winter"), (25, 20, -2, "Deep Winter"),
    (45, 10, -5, "True Winter"), (40, 12, -8, "True Winter"), (35, 14, -6, "True Winter"),
    (55, 25, 5, "Bright Winter"), (50, 28, 8, "Bright Winter"), (45, 30, 5, "Bright Winter")
]

# Helping function to get label for MST level
def get_mst_label(level_idx):
    if 0 <= level_idx < len(MST_LAB):
        return MST_LAB[level_idx][3]
    return "Unknown"
