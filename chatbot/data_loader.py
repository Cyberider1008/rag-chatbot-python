import json
import pandas as pd
from typing import List, Tuple

def load_data(file_path: str ="data/transactions.json") -> Tuple[List[str], pd.DataFrame]:
    """Load transaction data n convert to simple text."""
    try:

        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])  # convert to timestamp
        df["month"] = df["date"].dt.to_period("M").astype(str)  

        texts = []
        for _, row in df.iterrows():
            text = f"On {row['date'].date()}, {row['customer']} purchased a {row['product']} for ₹{row['amount']}."
            texts.append(text)
        
        return texts, df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")
