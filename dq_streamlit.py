import streamlit as st
import pandas as pd
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# --- SETUP ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]  # Secure key from Streamlit secrets

template = """
You are a data quality expert.

Convert the following plain English data quality rule into a YAML-formatted rule for a Python-based data quality engine.

YAML Format:
- rule_id: <unique_rule_id>
  description: <detailed_description>
  condition: <optional pandas query condition>  # optional
  check: <boolean pandas expression that returns True for valid rows>

Rule:
"{dq_rule}"

Return only the YAML block, no explanation.
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-3.5-turbo"
chain = prompt | llm

# --- FUNCTIONS ---
def append_rule_to_yaml(yaml_text, file_path="sample_rules.yaml"):
    try:
        new_rules = yaml.safe_load(yaml_text)
        if not isinstance(new_rules, list):
            new_rules = [new_rules]
        with open(file_path, "a") as f:
            yaml.dump(new_rules, f, sort_keys=False)
    except Exception as e:
        st.error(f"Failed to write to YAML file: {e}")

def load_rules(path="sample_rules.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def apply_rules(df, rules):
    results = []
    local_env = {"df": df, "pd": pd}

    for rule in rules:
        try:
            temp_df = df.copy()
            if "condition" in rule:
                mask = eval(rule["condition"], {}, local_env)
                temp_df = temp_df[mask]

            check_mask = eval(rule["check"], {}, {"df": temp_df, "pd": pd})
            failed = temp_df[~check_mask]

            results.append({
                "Rule ID": rule["rule_id"],
                "Description": rule["description"],
                "Violations": len(failed),
                "Violation %": round(len(failed) / len(df) * 100, 2)
            })

        except Exception as e:
            results.append({
                "Rule ID": rule.get("rule_id", "unknown"),
                "Description": f"Error: {str(e)}",
                "Violations": -1,
                "Violation %": 0
            })

    return pd.DataFrame(results)

# --- STREAMLIT APP ---
st.title("🧠 AI-Powered Data Quality Rule Engine")

# Load sample dataset
df = pd.read_csv("sample_data.csv")
st.subheader("📄 Sample Data")
st.dataframe(df)

# Rule Input
st.subheader("Enter Your Rule in Plain English")
user_rule = st.text_area("Describe your data quality rule (e.g., If KYCType is IDP8 then KYCNumber length should be 10)")

if st.button("Generate & Apply Rule"):
    if not user_rule.strip():
        st.warning("Please enter a rule.")
    else:
        # Generate YAML rule
        yaml_rule = chain.invoke({"dq_rule": user_rule}).content
        st.code(yaml_rule, language="yaml")

        # Append to file
        append_rule_to_yaml(yaml_rule)

        # Apply updated rules
        rules = load_rules()
        result = apply_rules(df, rules)

        st.subheader("📊 Data Quality Report")
        st.dataframe(result)
        st.success("Rule applied and report generated!")
  
