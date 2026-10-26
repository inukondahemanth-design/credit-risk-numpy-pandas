# Data Dictionary: German Credit Data

Source: [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). The data is from the 1970s, so money is in **DM (Deutsche Mark)**.

## Numeric columns (real numbers)
| Column | Meaning |
|---|---|
| duration | Loan length in **months** |
| credit_amount | Loan amount in **DM** |
| installment_rate | Monthly payment as % of disposable income (1 = low … 4 = high) |
| residence_since | Years living at current home (1–4) |
| age | Age in years |
| existing_credits | Number of loans already at this bank |
| num_dependents | Number of people the applicant supports financially |
| class | **Target**: 1 = good credit risk, 2 = bad credit risk |

## Coded columns (each code stands for a category)
| Column | Codes |
|---|---|
| checking_status | A11: < 0 DM · A12: 0–200 DM · A13: ≥ 200 DM · A14: no checking account |
| credit_history | A30: no credits taken / all paid back · A31: all credits at this bank paid back · A32: existing credits paid on time so far · A33: delays in the past · A34: critical account / credits at other banks |
| purpose | A40: new car · A41: used car · A42: furniture/equipment · A43: radio/TV · A44: appliances · A45: repairs · A46: education · A47: vacation · A48: retraining · A49: business · A410: other |
| savings | A61: < 100 DM · A62: 100–500 DM · A63: 500–1000 DM · A64: ≥ 1000 DM · A65: unknown / no savings |
| employment | A71: unemployed · A72: < 1 year · A73: 1–4 years · A74: 4–7 years · A75: ≥ 7 years |
| personal_status | A91: male, divorced/separated · A92: female, divorced/separated/married · A93: male, single · A94: male, married/widowed · A95: female, single |
| other_debtors | A101: none · A102: co-applicant · A103: guarantor |
| property | A121: real estate · A122: savings agreement / life insurance · A123: car or other · A124: unknown / none |
| other_installment_plans | A141: bank · A142: stores · A143: none |
| housing | A151: rent · A152: own · A153: for free |
| job | A171: unemployed / unskilled non-resident · A172: unskilled resident · A173: skilled employee · A174: management / self-employed / highly qualified |
| telephone | A191: none · A192: yes, registered |
| foreign_worker | A201: yes · A202: no |
