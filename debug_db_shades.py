from app import app, Foundation
with app.app_context():
    f_list = Foundation.query.filter(Foundation.brand=='Armani Beauty', Foundation.product.contains('Neo Nude')).limit(15).all()
    print("Name | L | a | b")
    print("-" * 20)
    for f in f_list:
        print(f"{f.name} | {f.l:.2f} | {f.a:.2f} | {f.b:.2f}")
