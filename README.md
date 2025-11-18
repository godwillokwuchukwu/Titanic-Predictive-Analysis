# Unveiling the Titanic’s Hidden Truths

### A Complete Data Science Case Study (EDA → ML Prediction → Streamlit App)

**By Okwuchukwu Godwill Tochukwu**
Data Scientist | Machine Learning Engineer | Data Storyteller

## Why I Built This Project (And Why It Still Matters in 2025)

The Titanic dataset has been analyzed a thousand times, but I wanted to do more than a typical Kaggle notebook.

My goals were to:

* Quantify *exactly* how age, class, gender, fare, and family size influence survival
* Build a machine learning model that predicts survival probability
* Create an interactive **Streamlit app** for real-time exploration
* Derive modern-day recommendations for the cruise/maritime industry

This project showcases end-to-end data science and product thinking:
**EDA → Feature Engineering → ML Modeling → Deployment → Storytelling.**

---

## Data Cleaning & Feature Engineering

*The part that quietly decides your model’s performance.*

**Dataset:** Titanic dataset from Seaborn (891 passengers)

### Key data challenges

* Missing Age values (~20%)
* 77% missing Cabin values
* Categorical columns needing clean encoding
* Extreme fare outliers
* Inconsistent column naming across sources

### What I did

```python
# Clean column names
df.rename(columns={
    'sibsp': 'Siblings_Spouses',
    'parch': 'Parents_Children'
})

# Create meaningful features
df['FamilySize'] = df['Siblings_Spouses'] + df['Parents_Children'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['AgeGroup'] = pd.cut(
    df['age'], 
    bins=[0,12,18,35,60,100],
    labels=['Child','Teen','Adult','Middle Age','Senior']
)

# Encode gender
df['Sex_code'] = df['sex'].map({'male': 1, 'female': 0})
```

**Impact:**
These improvements boosted the final model accuracy from **~73% → ~80%**.

---

## Exploratory Data Analysis: Where the Truth Revealed Itself

Using Plotly and Seaborn, I uncovered patterns that weren't just historical, they were human.

### **Top 5 Survival Insights**

1. **Women survived at 74.2% vs 18.9% for men**
   → Gender was the strongest survival factor.

2. **Class hierarchy was a matter of life and death**

   * 1st class: 63% survival
   * 2nd class: 47%
   * 3rd class: 24%

3. **Children under 12 had ~60% survival**
   → "Women and children first" wasn’t just a saying.

4. **Passengers from Cherbourg had the highest survival (55%)**
   → More first-class passengers boarded there.

5. **Best survival for families of 2–4**
   → Alone = low survival. Too large = chaos.

EDA gave this project a heartbeat — the story behind the numbers.

---

## Machine Learning: Predicting Survival

I focused on a **Logistic Regression** model for clarity and interpretability.

### Features used

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare

### Workflow

* Train/test split (80/20)
* StandardScaler
* LogisticRegression with balanced weights
* 5-fold cross-validation

### Results

* **Accuracy:** 80.4%
* **Fast, interpretable, deployable**

I saved the model with:

```python
joblib.dump(model, 'Titanic_model_lr.model')
```

And for instant Streamlit predictions, I implemented a lightweight manual logit function using the model coefficients.

---

## The Streamlit Dashboard (The Product)

Built with **Streamlit + Plotly**, the app includes:

* Sidebar filters for dynamic analysis
* 6 interactive visualizations
* Real-time survival probability calculator
* Key insights & interpretation
* Clean, modern UI
* Instant updates based on user input

This lets anyone — technical or not — explore survival patterns in seconds.

---

## What Modern Cruise Lines Can Learn (2025–2035)

The Titanic wasn’t just an accident, it was structural failure.

### Short-term (2025–2027)

* AI-assisted identification of vulnerable passengers
* Eliminating class-based evacuation bias
* Keep families together during emergencies

### Mid-term (2027–2030)

* Computer vision + wearables for tracking
* AI-driven evacuation routing
* Digital, gamified safety drills

### Long-term (2030–2035)

* Autonomous AI-powered lifeboats
* Optional pre-cruise risk profiling
* Satellite-linked global emergency response

History repeats itself, unless we learn from the data.

---

## Final Reflections

This project proves my ability to:

* Clean and interpret real-world messy data
* Build accurate, explainable ML models
* Develop full-stack analytics apps
* Communicate insights clearly
* Deliver production-quality solutions end-to-end

If your team needs someone who does more than write notebooks — someone who ships real, usable products — let’s talk.

---
**GitHub Repo:** (link in comments)
Notebook available on request.

---
