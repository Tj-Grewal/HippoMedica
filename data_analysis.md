[Link to Database explanation](https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names)

## Statistic: 768 examples
- "Not diabetes" (outcome = 0): 500
- "Diabetes" (outcome = 1): 268
- Note: all examples are of female patiens at least 21 years old of Pima Indian heritage

## Feature explanations:
   1. Pregnancies (int): number of times pregnant
   2. Glucose (?): plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. BloodPressure (int): diastolic blood pressure (mm Hg)
   4. SkinThickness (int): Triceps skin fold thickness (mm)
   5. Insulin (int): 2-Hour serum insulin (mu U/ml)
   6. BMI (float): Body mass index (weight in kg/(height in m)^2)
   7. DiabetesPedigreeFunction (float): likelihood of dieabetes based on family history (0->1 percentage)
   8. Age (int): age (years)
   9. Outcome: actual value diabetes/Not diabetes (0 or 1)

## Feature selection:
   1. Pregnancies: *keep raw* (relevant to diabetes risk, especially gestational, all patients are female)
   2. Glucose: *standardize* (direct link)
   3. BloodPressure: *standardize* (continuous value)
   4. SkinThickness: drop/standardize? (if one has diabetes, seem like there would be thickening of skins?)
   5. Insulin: didn't really understand this feature, couldn't find anything on this
   6. BMI: *standardize* (strong indicator)
   7. DiabetesPedigreeFunction: *standardize* (seems pretty relevant)
   8. Age: *standardize* (older = higher risk)
   9. Outcome: target value

**The choices above are up to discussion. Special attention required for SkinThickness and Insulin**