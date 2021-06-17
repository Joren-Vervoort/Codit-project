from utils.preprocessing import create_csv
from utils.feature_and_model_selection import concatenate_pd, undersampling, split_data, model_selection

# list of machines and list of added noise levels
machines = ["fan", "slider", "pump", "valve"]
dB_levels = ["-6", "0", "6"]

# loop to train models
for machine in machines:
    for dB_level in dB_levels:
        create_csv(machine, dB_level)

    df_merged = concatenate_pd(machine)
    under_sampled_data = undersampling(df_merged)
    X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(under_sampled_data)
    model_selection(X_train, X_test, X_valid, y_train, y_test, y_valid)

"""
If you want to test your saved model you can run the following function:

def model_prediction():
    model = input("Which model do you want to use to predict your .wav file?: ")

    loaded_model = pickle.load(open(filename, 'rb'))
    file = input("insert the path to your file: ")
    df=extract_features(file)

    X = df.drop(columns = ['normal(0)/abnormal(1)'])
    X_reduced=X[['melspectrogram', 'melspectrogram_sum', 'melspectrogram_std', 'mfcc',
        'rms', 'spectral_flatness ', 'spectral_rolloff']]
    y_pred= loaded_model.predict(X_reduced)
    
    if y_pred == 0:
        print("The machine is running normal.")
    else:
        print("The machine needs maintenance")
"""

