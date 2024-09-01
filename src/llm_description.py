import google.generativeai as genai

genai.configure(api_key="AIzaSyAx5ivzLvAFNQMYliCBF3zvSuOWFoONc_A")


def generate_description(dataset, commodity):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(
        f""" This is the data of vegetable market in nepal of {commodity}. Logically give the descriptive analysis of the given dataset of {commodity}
*Dataset Provided*:
- {dataset.groupby("Commodity").get_group(commodity)}

Please ensure that your analysis is clear, concise, and accurate.If additional information would be helpful.
                                      """
    )
    print(response.text)
    return response.text
