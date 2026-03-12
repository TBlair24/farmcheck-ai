# Exact class names from PlantVillage dataset folders
LABEL_MAP = {
    # PEPPER
    "Pepper__bell__Bacterial_spot":   {"domain": "agriculture", "indicator": "bacterial_infection", "compliant": False},
    "Pepper__bell__healthy":          {"domain": "agriculture", "indicator": "crop_healthy",        "compliant": True},

    # POTATO
    "Potato__Early_blight":           {"domain": "agriculture", "indicator": "fungal_blight",       "compliant": False},
    "Potato__healthy":                {"domain": "agriculture", "indicator": "crop_healthy",        "compliant": True},
    "Potato__Late_blight":            {"domain": "agriculture", "indicator": "fungal_blight",       "compliant": False},

    # TOMATO
    "Tomato__Target_Spot":            {"domain": "agriculture", "indicator": "leaf_disease",        "compliant": False},
    "Tomato__Tomato_mosaic_virus":    {"domain": "agriculture", "indicator": "viral_infection",     "compliant": False},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"domain": "agriculture", "indicator": "viral_infection", "compliant": False},
    "Tomato_Bacterial_spot":          {"domain": "agriculture", "indicator": "bacterial_infection", "compliant": False},
    "Tomato_Early_blight":            {"domain": "agriculture", "indicator": "fungal_blight",       "compliant": False},
    "Tomato_healthy":                 {"domain": "agriculture", "indicator": "crop_healthy",        "compliant": True},
    "Tomato_Late_blight":             {"domain": "agriculture", "indicator": "fungal_blight",       "compliant": False},
    "Tomato_Leaf_Mold":               {"domain": "agriculture", "indicator": "fungal_blight",       "compliant": False},
    "Tomato_Septoria_leaf_spot":      {"domain": "agriculture", "indicator": "leaf_disease",        "compliant": False},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"domain": "agriculture", "indicator": "pest_infestation", "compliant": False},
}

def get_compliance_label(class_name: str) -> dict:
    return LABEL_MAP.get(class_name, {
        "domain": "unknown",
        "indicator": "unclassified",
        "compliant": None
    })

def get_binary_label(class_name: str) -> int:
    """Returns 1 for compliant (healthy), 0 for non-compliant (diseased)"""
    info = get_compliance_label(class_name)
    return 1 if info["compliant"] else 0

if __name__ == "__main__":
    compliant = [k for k, v in LABEL_MAP.items() if v["compliant"]]
    non_compliant = [k for k, v in LABEL_MAP.items() if not v["compliant"]]
    
    print(f"\n✅ COMPLIANT classes ({len(compliant)}):")
    for c in compliant:
        print(f"   ✅ {c}")
    
    print(f"\n❌ NON-COMPLIANT classes ({len(non_compliant)}):")
    for c in non_compliant:
        print(f"   ❌ {c:55} → {LABEL_MAP[c]['indicator']}")
    
    print(f"\n📊 Summary: {len(LABEL_MAP)} total classes mapped")