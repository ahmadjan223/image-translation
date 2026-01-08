import json
import math

def calculate_pkr_price(sku_data):
    """
    Calculates the PKR price for a single SKU based on the PDF listing rules.
    """
    # --- Constants from PDF ---
    EXCHANGE_RATE = 40              # 
    FIRST_MILE_FIXED_YUAN = 3       # 
    MIDDLE_MILE_RATE = 2.842        # 
    PACKAGING_WEIGHT_G = 25         # [cite: 18]
    MARGIN_MULTIPLIER = 1.25        # 
    FIXED_MARKUP_PKR = 200          # 
    VOLUMETRIC_DIVISOR = 5000       # [cite: 20]
    VOLUMETRIC_THRESHOLD_G = 500    # [cite: 19]

    # --- Extract Data ---
    # Defaulting to 0 if key is missing to prevent errors
    price_yuan = sku_data.get('consignPrice', 0)
    actual_weight_g = sku_data.get('weight_g', 0)
    
    # Dimensions
    l = sku_data.get('length_cm', 0)
    w = sku_data.get('width_cm', 0)
    h = sku_data.get('height_cm', 0)

    # --- Step 1: First Mile Cost ---
    # Formula: (Product Price + 3) * 40
    first_mile_cost = (price_yuan + FIRST_MILE_FIXED_YUAN) * EXCHANGE_RATE

    # --- Step 2: Middle Mile Cost ---
    # Logic: Volumetric is only calculated if Actual Weight > 500g [cite: 80]
    if actual_weight_g > VOLUMETRIC_THRESHOLD_G:
        volumetric_weight = (l * w * h) / VOLUMETRIC_DIVISOR
        base_weight = max(actual_weight_g, volumetric_weight) # 
    else:
        base_weight = actual_weight_g

    # Add packaging weight to get Final Weight [cite: 18]
    final_weight = base_weight + PACKAGING_WEIGHT_G

    # Calculate cost
    middle_mile_cost = final_weight * MIDDLE_MILE_RATE

    # --- Step 3: Margin & Markup ---
    subtotal = first_mile_cost + middle_mile_cost
    
    # Apply 25% margin
    with_margin = subtotal * MARGIN_MULTIPLIER
    
    # Round up to nearest whole number (ceil) and add fixed markup [cite: 37]
    final_price = math.ceil(with_margin) + FIXED_MARKUP_PKR

    return final_price

def process_products(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Iterate through each product in the list
        for product in data:
            if 'validated_skus' in product:
                for sku in product['validated_skus']:
                    # Calculate price
                    pkr_price = calculate_pkr_price(sku)
                    
                    # Add new attribute to the SKU
                    sku['pkr_price'] = pkr_price
                    
                    # Optional: Print for verification
                    print(f"SKU {sku.get('sku_index')}: Weight={sku.get('weight_g')}g | "
                          f"Price={sku.get('consignPrice')} Yuan -> Final PKR: {pkr_price}")

        # Save the modified data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"\nSuccess! Processed data saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execution ---
# Replace 'modeloutput withprice.json' with your actual filename if different
input_filename = 'apiResponse.json'
output_filename = 'updated_product_prices.json'

if __name__ == "__main__":
    process_products(input_filename, output_filename)