import csv
import cv2
import numpy as np
import os
import pandas as pd

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def get_brand_tier(brand):
    tier = 'Other'
    brand_lower = brand.lower()
    if any(b in brand_lower for b in ['fenty', 'armani', 'estee', 'estée', 'lancome', 'lancôme', 'charlotte tilbury', 'nars']):
        tier = 'High-End'
    elif any(b in brand_lower for b in ['maybelline', 'nyx', 'l\'oreal', 'loreal', 'revlon', 'covergirl']):
        tier = 'Drugstore'
    return tier

def sync():
    from app import app, db, Foundation, cv2_to_std_lab
    sources = [
        {
            'path': 'all_shades/allShades.csv',
            'type': 'legacy'
        },
        {
            'path': 'concealer/all.csv',
            'type': 'new_concealer'
        }
    ]

    with app.app_context():
        print("Dropping and recreating Foundation table...")
        Foundation.__table__.drop(db.engine)
        Foundation.__table__.create(db.engine)

        total_count = 0

        for source in sources:
            csv_path = source['path']
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found. Skipping...")
                continue

            print(f"Syncing from {csv_path}...")

            if source['type'] == 'legacy':
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        brand = row['brand']
                        product = row['product']
                        name_raw = str(row.get('name', '')).strip()
                        spec_raw = str(row.get('specific', '')).strip()

                        if spec_raw and spec_raw.lower() not in ['nan', 'na', 'n/a', 'none', '']:
                            name = spec_raw
                        elif name_raw and name_raw.lower() not in ['nan', 'na', 'n/a', 'none', '']:
                            name = name_raw
                        elif ' - ' in product:
                            name = product.split(' - ')[-1]
                        else:
                            name = "Shade"

                        hex_val = row['hex']
                        hue = row['hue']
                        image_url = row['imgSrc']

                        category = 'Foundation'
                        if 'concealer' in product.lower() or 'concealer' in row.get('description', '').lower():
                            category = 'Concealer'

                        try:
                            rgb = hex_to_rgb(hex_val)
                            bgr = np.uint8([[list(rgb[::-1])]])
                            lab_cv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
                            l_std, a_std, b_std = cv2_to_std_lab(float(lab_cv2[0]), float(lab_cv2[1]), float(lab_cv2[2]))
                        except:
                            continue

                        f_obj = Foundation(
                            brand=brand, product=product, name=name, hex_code=hex_val,
                            l=l_std, a=a_std, b=b_std, category=category,
                            brand_tier=get_brand_tier(brand), hue=hue, image_url=image_url
                        )
                        db.session.add(f_obj)
                        total_count += 1
                        if total_count % 500 == 0:
                            db.session.commit()
                            print(f"Synced {total_count} shades total...")

            elif source['type'] == 'new_concealer':
                df = pd.read_csv(csv_path)
                df.columns = [c.strip("'").strip().lower() for c in df.columns]
                concealers_df = df[df['product_name'].str.contains('conceal', case=False, na=False)].copy()
                print(f"Found {len(concealers_df)} concealer shades in new dataset.")

                for _, row in concealers_df.iterrows():
                    brand = row['brand_name']
                    product = row['product_name']
                    name = row['shade_name']
                    hex_val = row['hex']
                    l_val = float(row['l'])
                    a_val = float(row['a'])
                    b_val = float(row['b'])

                    f_obj = Foundation(

                        brand=brand, product=product, name=name, hex_code=hex_val,
                        l=l_val, a=a_val, b=b_val, category='Concealer',
                        brand_tier=get_brand_tier(brand), hue='Neutral', image_url=''
                    )
                    db.session.add(f_obj)
                    total_count += 1
                    if total_count % 500 == 0:
                        db.session.commit()
                        print(f"Synced {total_count} shades total...")

        db.session.commit()
        print(f"Sync complete. Total shades: {total_count}")

if __name__ == "__main__":
    sync()
