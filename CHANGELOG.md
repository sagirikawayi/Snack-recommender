📝 Update Log (Changelog)
[v1.1.0] - 2026-04-03
"The Precision & Standards Update"

🚀 New Features
Precise Nutrient Control: Added Number Input fields synchronized with Sliders for exact calorie and protein targeting.

EU Nutritional Standards Guidance: Integrated real-time feedback cards based on EU Regulation (EC) No 1924/2006 to categorize food:

Energy: Low Energy (≤120kcal), Standard, High Energy.

Protein: Low Protein, Source of Protein (≥5g), High Protein (≥12g).

Dual-Binding UI: Implemented st.session_state and callback functions to ensure seamless synchronization between sliders and manual input boxes.

🛠️ Technical Improvements
Language Localization: Migrated the entire User Interface (UI) to English for international professional standards.

Refactored Logic: Optimized the KNN feature vector scaling to align with the new precise input system.

Project Restructuring: Renamed the entry point from test.py to Food.py for better repository clarity.

🐞 Bug Fixes
Fixed the DuplicateWidgetID error caused by shared keys between input components.

Corrected the Euclidean distance calculation to handle edge cases with zero-variance inputs.

[v1.0.0] - 2026-03-25
"Initial Release: AI Food Recommender"

Base KNN algorithm implementation using scikit-learn.

Initial UI with Streamlit.

Data integration for healthy_foods_database.csv.
