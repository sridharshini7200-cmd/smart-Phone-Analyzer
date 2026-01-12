import pandas as pd
from google.adk.agents.llm_agent import Agent
from typing import List


try:
    _DF = pd.read_csv('/content/smartphones.csv')

    _DF = _DF.dropna(subset=['model', 'price', 'avg_rating', 'processor_brand', 'ram_capacity', 'battery_capacity'])
except FileNotFoundError:
    print("Error: 'smartphones.csv' not found. Agent tools will not function correctly.")
    _DF = pd.DataFrame()




def review_smartphone(model_name: str) -> str:
    """
    Provides a detailed review and summary of the key specifications for a specific smartphone model.
    Looks up price, rating, processor, RAM, and battery for the specified model.
    """
    if _DF.empty:
        return "Error: Smartphone data is not loaded."

    # Find the row containing the model name (case-insensitive)
    data = _DF[_DF['model'].str.contains(model_name, case=False, na=False)]

    if data.empty:
        return f"Model '{model_name}' not found in the database."

    # Use the first match
    data = data.iloc[0]

    review = (
        f"**Review for {data['brand_name']} {data['model']}**\n"
        f"Price: ${data['price']}\n"
        f"Average Rating: {data['avg_rating']}/10\n"
        f"Processor: {data['processor_brand']} series\n"
        f"RAM/Storage: {data['ram_capacity']}GB RAM / {data['internal_memory']}GB Storage\n"
        f"Battery: {data['battery_capacity']}mAh\n"
        f"Verdict: A solid option with a {data['processor_brand']} processor and high {data['ram_capacity']}GB RAM, making it good for performance at its price point."
    )
    return review


def compare_smartphones(model1_name: str, model2_name: str) -> str:
    """
    Compares the main specifications (price, rating, processor, RAM, battery) of two smartphone models.
    """
    if _DF.empty:
        return "Error: Smartphone data is not loaded."

    model1 = _DF[_DF['model'].str.contains(model1_name, case=False, na=False)]
    model2 = _DF[_DF['model'].str.contains(model2_name, case=False, na=False)]

    if model1.empty or model2.empty:
        missing = []
        if model1.empty: missing.append(model1_name)
        if model2.empty: missing.append(model2_name)
        return f"Could not find model(s): {', '.join(missing)} for comparison."

    data1 = model1.iloc[0]
    data2 = model2.iloc[0]

    comparison = (
        f"**Side-by-Side Comparison:**\n"
        f"| Feature | {data1['model']} | {data2['model']} |\n"
        f"| :--- | :--- | :--- |\n"
        f"| Price | ${data1['price']} | ${data2['price']} |\n"
        f"| Avg. Rating | {data1['avg_rating']} | {data2['avg_rating']} |\n"
        f"| Processor | {data1['processor_brand']} | {data2['processor_brand']} |\n"
        f"| RAM | {data1['ram_capacity']}GB | {data2['ram_capacity']}GB |\n"
        f"| Battery | {data1['battery_capacity']}mAh | {data2['battery_capacity']}mAh |\n"
        f"| 5G | {'Yes' if data1['5G_or_not'] == 1 else 'No'} | {'Yes' if data2['5G_or_not'] == 1 else 'No'} |"
    )
    return comparison


def recommend_smartphones(max_price: int, min_rating: float, required_5g: bool) -> List[str]:
    """
    Recommends a list of up to five smartphone models based on a maximum price, minimum average rating, and 5G requirement.
    """
    if _DF.empty:
        return ["Error: Smartphone data is not loaded."]

    # Convert required_5g to integer 0 or 1 for filtering
    five_g_filter = 1 if required_5g else 0

    # Filter the DataFrame
    recommendations_df = _DF[
        (_DF['price'] <= max_price) &
        (_DF['avg_rating'] >= min_rating) &
        (_DF['5G_or_not'] >= five_g_filter)
    ]

    # Sort by rating (highest first) and price (lowest first)
    recommendations_df = recommendations_df.sort_values(by=['avg_rating', 'price'], ascending=[False, True])

    # Get the top 5 models
    top_models = recommendations_df.head(5)[['model', 'price', 'avg_rating']].to_dict('records')

    if not top_models:
        return [f"No smartphones found matching the criteria (Price: ${max_price}, Rating: {min_rating}, 5G: {required_5g})."]

    result_list = ["Based on your preferences, here are the top recommended smartphones:"]
    for model in top_models:
        result_list.append(f"- {model['model']} (Rating: {model['avg_rating']}, Price: ${model['price']})")

    return result_list



root_agent = Agent(
    name="smartphone_data_analyst",
    model="gemini-2.5-flash",
    description="A multi-tool agent for querying and analyzing smartphone specifications from a loaded dataset.",
    instruction = (
    "You are an expert Smartphone Recommendation Analyst.\n"
    "Use ONLY the provided tools to respond to user queries related to smartphone specifications, "
    "reviews, comparisons, pricing, ratings, and 5G or other technical capabilities.\n"
    "When a query requires factual smartphone data or structured analysis, invoke the appropriate tool "
    "(review_smartphone, compare_smartphones, or recommend_smartphones).\n"
    "For all other general, non–smartphone-related questions, respond directly using your own reasoning "
    "without using any tools.\n"
    "Ensure your responses remain accurate, concise, and tailored to the user's needs."
     ),
     tools=[
        review_smartphone,
        compare_smartphones,
        recommend_smartphones
      ],
)
