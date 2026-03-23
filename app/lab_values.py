def lab_value(blood_sugar, hemoglobin):
    context = []
    lab_suspected_diseases = []
    if blood_sugar and blood_sugar > 140:
        context.append("Blood sugar is above normal range")
        lab_suspected_diseases.append("Diabetes")
    if hemoglobin and hemoglobin < 12:
        context.append("Hemoglobin level is below normal range.")
        lab_suspected_diseases.append("Anemia")
    if not context:
        context.append("No abnormal lab values are detected.")

    return {
        "text": ". ".join(context),
        "hints": lab_suspected_diseases
    }
