import json

def load_data(file_path="data/transactions.json"):
    """Load transaction data n convert to simple text."""
    try:

        with open(file_path, "r") as f:
            data = json.load(f)

        texts = []
        for t in data:
            text = f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
            texts.append(text)
        return texts
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")
