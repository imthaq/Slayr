import os
import time

def reset_db():
    paths = [
        os.path.join(os.getcwd(), "instance", "slayr.db"),
        os.path.join(os.getcwd(), "slayr.db")
    ]
    
    print("Attempting to reset database...")
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Deleted: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")
                print("The file might be in use. Please ensure the Flask app is STOPPED.")
        else:
            print(f"Not found (clean): {path}")

if __name__ == "__main__":
    reset_db()
