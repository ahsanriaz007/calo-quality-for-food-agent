import pandas as pd
from groq import Groq

# 1. Initialize the Groq client
# IMPORTANT: Replace the string below with your actual Groq API key!
api_key = ""

client = Groq(api_key=GROQ_API_KEY)


def generate_action_plan(incident_text):
  """Uses Groq's Llama model to generate a fast, actionable summary."""

  # Fallback template if no API key is provided
  if GROQ_API_KEY == "gsk_YOUR_API_KEY_HERE":
    return (
        f"TEMPLATE FALLBACK: Investigate issue '{incident_text}'. "
        "Recommend shift lead review."
    )

  prompt = f"""
    You are an expert Kitchen Operations Manager for Calo, a premium meal prep service.
    Review the following operational incident or customer complaint.
    Write a strict, 2-sentence maximum action plan for the floor staff to resolve this.
    Be direct, professional, and focus on food safety or logistics.
    
    Incident Details: {incident_text}
    """

  try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",  # High-quality reasoning model
        temperature=0.2,  # Keep responses consistent and professional
    )
    return response.choices[0].message.content.strip()

  except Exception as e:
    return f"API Error: {str(e)}"


if __name__ == "__main__":
  print("Testing the LLM Summarizer...")

  # Test Case 1: A Kitchen Anomaly
  kitchen_issue = "High Fridge Temp (7.5°C > 6.0°C) | Prep Delay (35 mins > 30 mins) in Riyadh-Central"
  print("\n--- Test 1: Kitchen Anomaly ---")
  print("Input:", kitchen_issue)
  print("AI Plan:", generate_action_plan(kitchen_issue))

  # Test Case 2: A Customer Complaint
  customer_complaint = "Found a piece of hard plastic in my chicken and rice."
  print("\n--- Test 2: Customer Complaint ---")
  print("Input:", customer_complaint)
  print("AI Plan:", generate_action_plan(customer_complaint))