from preprocessing import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor

import pickle

def preprocess(csv_url = "D:/NBA-Predict/matches_w_player_stats.csv"):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.rslt)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_home_team_point(csv_url ="D:/NBA-Predict/matches_w_player_stats.csv"):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.teamPTS)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_away_team_point(csv_url ="D:/NBA-Predict/matches_w_player_stats.csv"):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.opptPTS)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_home_q1_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.teamPTS1)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_home_q2_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.teamPTS2)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_home_q3_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.teamPTS3)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_home_q4_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.teamPTS4)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_away_q1_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.opptPTS1)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_away_q2_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.opptPTS2)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_away_q3_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.opptPTS3)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def preprocess_away_q4_point(csv_url):
    df = pd.read_csv(csv_url)  # read data
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    df = old_to_new_team_abbrs(df)
    # players = get_unique_player_list(df)  # get all player list
    # team_abbrs = get_unique_team_abbr(df)  # get all team names
    df = drop_some_columns(df)
    df = clean_nan_values(df)
    Y = np.array(df.opptPTS4)
    encoded,dict1,dict2,dict3 = label_train_data(df)
    scalered ,ss= standart_scaler_all_data(encoded)
    df = onehotencoder_all_data(scalered)
    return df,Y
def split_data_by_result(df,result_index=60):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    print(Y)
    return X,Y
def split_data_home_point(df, result_index=62):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_away_point(df, result_index=67):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_home_q1_point(df, result_index=63):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_home_q2_point(df, result_index=64):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_home_q3_point(df, result_index=65):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_home_q4_point(df, result_index=66):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_away_q1_point(df, result_index=68):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_away_q2_point(df, result_index=69):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_away_q3_point(df, result_index=70):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y
def split_data_away_q4_point(df, result_index=71):
    X = df.drop(columns=result_index)
    Y = df[result_index]
    return X, Y

def build_classifier_model(X_train,Y_train,X_test,Y_test):
    # model = XGBClassifier(n_estimators=5000,nthread=4,seed=42,reg_lambda=0.95,reg_alpha=0.45,tree_method="gpu_hist",max_depth=3,objective="binary:logistic")
    model = XGBClassifier(tree_method="gpu_hist",nthread=4,n_estimators=20000)

    model.fit(X_train, Y_train,eval_set=[(X_test,Y_test)],eval_metric="error",early_stopping_rounds=42)
    pickle.dump(model,open("D:/NBA-Predict/model/xgb_classifier.pkl","wb"))
    print(model.score(X_test,Y_test))
    print(model.classes_)

    return model , model.score(X_test,Y_test)

def build_class_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess(csv_url)

    X,_ = split_data_by_result(df=df)
    X_train,X_test,Y_train,Y_test = data_split(X,Y)
    model_ , score__ = build_classifier_model(X_train,Y_train,X_test,Y_test)
    return model_,X_train,X_test,Y_train,Y_test

def build_regressor_model(X_train, Y_train, X_test, Y_test, name):
    # model = XGBRegressor(learning_rate=0.15,gamma=0,reg_lambda=0.01,max_delta_step=0, max_depth=3,n_estimators=10000,
    #                      min_child_weight=1,nthread=4,tree_method="gpu_hist")
    model = XGBRegressor(tree_method="gpu_hist",n_estimators=20000,nthread=4)
    model.fit(X_train, Y_train,eval_metric="rmse",eval_set=[(X_test,Y_test)],early_stopping_rounds=20)
    pickle.dump(model, open(name, "wb"))
    print(model.score(X_test,Y_test))
    return model, model.score(X_test, Y_test)
def build_home_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_home_team_point(csv_url)
    X,_ = split_data_home_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,
                                             name="D:/NBA-Predict/model/teams/home_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_away_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_away_team_point(csv_url)
    X,_ = split_data_away_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,name="D:/NBA-Predict/model/teams/away_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_home_q1_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_home_q1_point(csv_url)
    X,_ = split_data_home_q1_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/home_q1_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_home_q2_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_home_q2_point(csv_url)
    X,_ = split_data_home_q2_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/home_q2_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_home_q3_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_home_q3_point(csv_url)
    X,_ = split_data_home_q3_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/home_q3_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_home_q4_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_home_q4_point(csv_url)
    X,_ = split_data_home_q4_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/home_q4_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_away_q1_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_away_q1_point(csv_url)
    X,_ = split_data_away_q1_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/away_q1_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_away_q2_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_away_q2_point(csv_url)
    X,_ = split_data_away_q2_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/away_q2_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_away_q3_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_away_q3_point(csv_url)
    X,_ = split_data_away_q3_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/away_q3_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def build_away_q4_point_predict_model(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df,Y=preprocess_away_q4_point(csv_url)
    X,_ = split_data_away_q4_point(df=df)
    X_train,X_test,Y_train,Y_test = data_split_regresyon(X,Y)
    model_ , score__ = build_regressor_model(X_train,Y_train,X_test,Y_test,name="D:/NBA-Predict/model/teams/away_q4_point_model.pkl")
    return model_,X_train,X_test,Y_train,Y_test
def get_result(X, model):
    return model.predict_proba(X)
def get_point_result(X, model):
    return model.predict(X)
def get_points(X,model):
    return model.predict(X) ,model.predict_proba(X)
def data_split(X,Y,test_size=0.1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return X_train,X_test,Y_train,Y_test
def data_split_regresyon(X,Y,test_size=0.1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return X_train,X_test,Y_train,Y_test
def predict_match_result(data):
    model = pickle.load(open("D:/NBA-Predict/model/xgb_classifier.pkl","rb"))
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    print(df.head())
    X, Y = split_data_by_result(df=df,result_index=60)
    return get_result(X, model),int(Y[0])
def predict_home_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/home_point_model.pkl","rb"))
    Y = np.array(data.teamPTS)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_home_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_away_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/away_point_model.pkl","rb"))
    Y = np.array(data.opptPTS)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_away_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_home_q1_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/home_q1_point_model.pkl","rb"))
    Y = np.array(data.teamPTS1)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_home_q1_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_home_q2_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/home_q2_point_model.pkl","rb"))
    Y = np.array(data.teamPTS2)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_home_q2_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_home_q3_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/home_q3_point_model.pkl","rb"))
    Y = np.array(data.teamPTS3)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_home_q3_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_home_q4_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/home_q4_point_model.pkl","rb"))
    Y = np.array(data.teamPTS4)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_home_q4_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_away_q1_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/away_q1_point_model.pkl","rb"))
    Y = np.array(data.opptPTS1)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_away_q1_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_away_q2_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/away_q2_point_model.pkl","rb"))
    Y = np.array(data.opptPTS2)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_away_q2_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_away_q3_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/away_q3_point_model.pkl","rb"))
    Y = np.array(data.opptPTS3)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_away_q3_point(df=df)
    return get_point_result(X, model),int(Y[0])
def predict_away_q4_point(data):
    model = pickle.load(open("D:/NBA-Predict/model/teams/away_q4_point_model.pkl","rb"))
    Y = np.array(data.opptPTS4)
    encoded= label_test_data(data)
    scalered = standart_scaler_test_data(encoded)
    df = onehotencoder_test_data(scalered)
    X, _ = split_data_away_q4_point(df=df)
    return get_point_result(X, model),int(Y[0])
def build_all_in_one(csv_url="D:/NBA-Predict/matches_w_player_stats.csv"):
    df = pd.read_csv("D:/NBA-Predict/test_data/GSWvsHOU_test_data.csv", index_col=0)
    df.drop("index", axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df_2 = df.copy()
    df_3 = df.copy()
    df_4 = df.copy()
    df_5 = df.copy()
    df_6 = df.copy()
    df_7 = df.copy()
    df_8 = df.copy()
    df_9 = df.copy()
    df_10 = df.copy()
    df_11 = df.copy()

    model_, X_train, X_test, Y_train, Y_test = build_class_model(csv_url=csv_url)
    print(predict_match_result(df_11))

    # model_, X_train, X_test, Y_train, Y_test = build_home_q1_point_predict_model(csv_url=csv_url)
    # print(predict_home_q1_point(df_5))
    # model_, X_train, X_test, Y_train, Y_test = build_home_q2_point_predict_model(csv_url=csv_url)
    # print(predict_home_q2_point(df_6))
    # model_, X_train, X_test, Y_train, Y_test = build_home_q3_point_predict_model(csv_url=csv_url)
    # print(predict_home_q3_point(df_7))
    # model_, X_train, X_test, Y_train, Y_test = build_home_q4_point_predict_model(csv_url=csv_url)
    # print(predict_home_q4_point(df_8))
    # model_, X_train, X_test, Y_train, Y_test = build_away_q1_point_predict_model(csv_url=csv_url)
    # print(predict_away_q1_point(df))
    # model_, X_train, X_test, Y_train, Y_test = build_away_q2_point_predict_model(csv_url=csv_url)
    # print(predict_away_q2_point(df_2))
    # model_, X_train, X_test, Y_train, Y_test = build_away_q3_point_predict_model(csv_url=csv_url)
    # print(predict_away_q3_point(df_3))
    # model_, X_train, X_test, Y_train, Y_test = build_away_q4_point_predict_model(csv_url=csv_url)
    # print(predict_away_q4_point(df_4))
    # model_, X_train, X_test, Y_train, Y_test = build_home_point_predict_model(csv_url=csv_url)
    # print(predict_home_point(df_9))
    # model_, X_train, X_test, Y_train, Y_test = build_away_point_predict_model(csv_url=csv_url)
    # print(predict_away_point(df_10))


if __name__ == '__main__':
    build_all_in_one()



