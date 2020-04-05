import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

MAX_YEARS = 30
x = sorted([0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 30.0])
DOMAIN = list(range(1, MAX_YEARS+1))
EXPERIENCE_LEVEL_TRANSLATIONS = {
    'Less than 1 year': 0.0,
    'More than 50 years': 50,
    '0-2 years': 1.0,
    '3-5 years': 4.0,
    '6-8 years': 7.0,
    '9-11 years': 10.0,
    '12-14 years': 13.0,
    '15-17 years': 16.0,
    '18-20 years': 19.0,
    '21-23 years': 22.0,
    '24-26 years': 25.0,
    '27-29 years': 28.0,
    '30 or more years': 30.0
}

EDUCATION_TYPES = [
    "Associate degree",
    "Bachelor’s degree (BA, BS, B.Eng., etc.)",
    "Master’s degree (MA, MS, M.Eng., MBA, etc.)",
    "Other doctoral degree (Ph.D, Ed.D., etc.)"
]

def get_united_states_professionals_2018():
    df = pd.read_csv("2018/survey_results_public.zip")
    df = df[(df["Country"] == "United States") & (df["CurrencySymbol"] == "USD")]
    df = df.rename(columns={"FormalEducation": "EdLevel", "YearsCodingProf": "YearsCodePro", "ConvertedSalary": "ConvertedComp"})
    return df[["EdLevel", "YearsCodePro", "ConvertedComp"]]

def get_united_states_professionals_2019():
    survey_results_2019 = pd.read_csv("2019/survey_results_public.zip")
    professionals_2019 = survey_results_2019[(survey_results_2019["Country"] == "United States") & (survey_results_2019["CurrencySymbol"] == "USD")]
    return professionals_2019[["EdLevel", "YearsCodePro", "ConvertedComp"]]

def get_united_states_professionals():
    professionals_2018 = get_united_states_professionals_2018()
    professionals_2019 = get_united_states_professionals_2019()
    return pd.concat([professionals_2018, professionals_2019])

def get_logarithmic_points(coefficients):
    return np.array([np.log(x_)*coefficients[0] + coefficients[1] for x_ in DOMAIN])

def normalize_to_size(y, size):
    """
    Uses forward and then backward filling to stretch the array to the given size.
    """
    temp_array = []
    most_recent = np.nan
    for i in range(0,size):
        temp_array.append(y[i] if (y[i] and not math.isnan(y[i])) else most_recent)
        most_recent = temp_array[i]
    most_recent = y[-1]
    temp_array.reverse()
    y.reverse()
    for i in range(0,size):
        temp_array[i] = y[i] if (y[i] and not math.isnan(y[i])) else most_recent
        most_recent = temp_array[i]
    temp_array.reverse()
    return temp_array

if __name__ == "__main__":
    us_professionals = get_united_states_professionals()
    us_professionals["YearsCodePro"] = pd.to_numeric(us_professionals["YearsCodePro"].apply(lambda x: EXPERIENCE_LEVEL_TRANSLATIONS[x] if x in EXPERIENCE_LEVEL_TRANSLATIONS else x))
    max_y = 0
    for education_type in EDUCATION_TYPES:
        population = us_professionals[us_professionals["EdLevel"] == education_type]
        years_worked = pd.to_numeric(population["YearsCodePro"].dropna())
        comps = pd.to_numeric(population["ConvertedComp"].dropna())
        median_annual_compensation = np.median(comps)
        y = []
        for year in range(1, MAX_YEARS+1):
            if year in x:
                compensation_after_years = np.median(population[population["YearsCodePro"] == year]["ConvertedComp"].dropna())
                y.append(compensation_after_years)
            else:
                y.append(float('nan')) # will forward-fill these nan values
        y = normalize_to_size(y, MAX_YEARS)
        log_fit_parameters = np.polyfit(np.log(DOMAIN), y, 1)
        plt.plot(DOMAIN, get_logarithmic_points(log_fit_parameters), label=education_type)
        # plt.plot(DOMAIN, y, label=education_type + " Raw Points")
        max_y = max(max_y, np.max(get_logarithmic_points(log_fit_parameters)))
    plt.legend()
    plt.xlabel("Years Of Professional Coding")
    plt.ylabel("Annual Compensation (USD)")
    plt.title("Coding Pros' Comp by Level of Ed. (2018 & 2019)")
    plt.grid()
    plt.yticks(np.arange(0, 170000, 10000))
    plt.savefig("ed_level_compensation_per_year.png")