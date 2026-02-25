from app import app, Foundation, db
with app.app_context():
    concealers = Foundation.query.filter_by(category='Concealer').all()
    brands = sorted(list(set([c.brand for c in concealers])))
    print(f"Total Concealer Shades: {len(concealers)}")
    print(f"Total Unique Concealer Brands: {len(brands)}")
    
    # Check specific brands requested previously
    armani = Foundation.query.filter(Foundation.category=='Concealer', Foundation.brand.contains('Armani')).count()
    fenty = Foundation.query.filter(Foundation.category=='Concealer', Foundation.brand.contains('Fenty')).count()
    estee = Foundation.query.filter(Foundation.category=='Concealer', Foundation.brand.contains('Estée')).count()
    
    print(f"Armani Concealers: {armani}")
    print(f"Fenty Concealers: {fenty}")
    print(f"Estée Lauder Concealers: {estee}")
    
    print("\nSample Concealer Brands:")
    print(", ".join(brands[:20]))
