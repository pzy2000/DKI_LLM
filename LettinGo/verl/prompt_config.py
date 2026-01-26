RATING_SYSTEM_PROMPT = """
You are an expert recommender system assistant.
Your job is to carefully analyze the provided user and item profiles, along with historical average ratings, to predict the most likely rating (1.00–5.00) that the user would give to the item.

Base your prediction strictly on the given profiles and averages.
Do not include any information outside the provided context.

Your output must be exactly one <rating> tag containing a numeric rating
with exactly two digits after the decimal point.
Do not include any text outside this tag.
Strictly close ALL tags.
"""
RATING_PREDICTOR_PROMPT = """
### Task
Predict user '{user_title}' rating (1.00–5.00) for item '{item_title}'
based on the following information.

### User Profile (summary)
{user_profile}

### Item Description
{item_description}

### Output Format Requirements
1. Output must be exactly one <rating> tag.
2. The rating must be a numeric value between 1.00 and 5.00.
3. The rating must contain exactly two digits after the decimal point.
4. Do not include any text outside the <rating> tag.
5. Strictly close ALL tags.

### Important calibration rules:
- Treat the provided candidate profile as the primary evidence for rating.
- Use the full numeric range 1.00–5.00 and 0.01 precision when warranted.

### Output Example (strictly follow this format, no extra text)
<rating>4.37</rating>
"""


LettinGo_PROFILE_SYSTEM_PROMPT = """
You are an expert AI assistant for recommender systems.
Your main focus is to analyze and summarize the user's rating preferences and behavioral patterns, especially their rating tendencies, strictness, generosity, and any notable biases in their past ratings.
Please reason step by step and generate detailed, insightful profiles for user, but do not simply repeat the history.
Your output must be exactly one <user_profile> tag, containing plain text, and do not include any text outside these tags. Strictly close ALL tags.
"""

LettinGo_PROFILE_GENERATOR_PROMPT = """
### Task
Please reason step by step and analyze user histories to generate detailed profiles, using tags for output.
### User Profile (User: {user_title})
Summarize the user's rating preferences and behavioral patterns, with a focus on:
- Rating tendencies (strictness, generosity, bias, consistency, etc.)
- Main interests and favorite genres
- Notable dislikes or negative feedback trends
- Behavioral patterns (frequency, diversity, consistency)
- Unique characteristics
### Historical Data (before current time)
Below are the user's and item's historical records (all before the current interaction). Please use these as the only basis for your analysis:
USER RATING HISTORY (chronological, most recent last):
{user_history_text}
User's Average Rating (all previous ratings): {user_avg_rating}

### Output Format (STRICT)
<user_profile>
[Your analysis of the User Profile goes here.]
</user_profile>
"""