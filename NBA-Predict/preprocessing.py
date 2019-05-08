import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder

def get_golden_feature_list(df,threshold=.5):
    """

    :param df: Main Data csv
    :param threshold: Your Lower Bound of Correlation
    :return: Golden Feature List
    """
    df_numeric = df.select_dtypes(include=["float64","int64"])
    df_corr = df_numeric.corr()
    df_corr = df_corr.iloc[2,:-1]
    golden_features_list = df_corr[abs(df_corr) >= threshold].sort_values(ascending=False)
    return golden_features_list
def get_worst_feature_list(df,threshold=.5):
    """

    :param df: Main Dataframe
    :param threshold: Your Upper Bound
    :return: Worst Feature List
    """
    df_numeric = df.select_dtypes(include=["float64","int64"])
    df_corr = df_numeric.corr()
    df_corr = df_corr.iloc[2,:-1]
    golden_features_list = df_corr[abs(df_corr) <= threshold].sort_values(ascending=False)
    return golden_features_list
def remove_whitespaces_in_df_columns(df):
    """
    :param df:Main Dataframe
    :return: Cleaned Columns
    """
    df.columns = df.columns.str.replace(' ', '')
    return df.columns
def get_column_names():
    """
    :return: Column names of Data.csv
    """
    feature_names_of_players = ["PLAYER_NAME", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
                                "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS"]

    feature_names_of_matchups = ["gameId", "teamAbbr", "opptAbbr", "rslt", "teamMin", "teamPTS", "teamPTS1", "teamPTS2",
                                 "teamPTS3", "teamPTS4", "opptPTS",
                                 "opptPTS1", "opptPTS2", "opptPTS3", "opptPTS4",
                                 "teamFGM", "teamFGA", "teamFG", "team3PM", "team3PA", "team3PCT", "teamFTM", "teamFTA",
                                 "teamFTC", "teamORB", "teamDRB", "teamREB", "teamAST", "teamSTL"
        , "teamBLK", "teamTO", "teamPF", "team2P", "teamTS", "teamEFG", "teamPPS", "teamFIC", "teamFIC40", "teamOrtg",
                                 "teamDrtg", "teamPlay",
                                 "opptMin", "opptFGM", "opptFGA", "opptFG", "oppt3PM", "oppt3PA", "oppt3PCT", "opptFTM",
                                 "opptFTA", "opptFTC", "opptORB", "opptDRB", "opptREB", "opptAST", "opptSTL"
        , "opptBLK", "opptTO", "opptPF", "oppt2P", "opptTS", "opptEFG", "opptPPS", "opptFIC", "opptFIC40", "opptOrtg",
                                 "opptDrtg", "opptPlay", "poss", "pace"]

    team_features = ["gameId", "teamAbbr", "opptAbbr", "rslt", "teamMin", "teamPTS", "teamPTS1", "teamPTS2", "teamPTS3",
                     "teamPTS4", "opptPTS",
                     "opptPTS1", "opptPTS2", "opptPTS3", "opptPTS4",
                     "teamFGM", "teamFGA", "teamFG", "team3PM", "team3PA", "team3PCT", "teamFTM", "teamFTA", "teamFTC",
                     "teamORB", "teamDRB", "teamREB", "teamAST", "teamSTL"
        , "teamBLK", "teamTO", "teamPF", "team2P", "teamTS", "teamEFG", "teamPPS", "teamFIC", "teamFIC40", "teamOrtg",
                     "teamDrtg", "teamPlay"]
    team_features = team_features + feature_names_of_players * 11

    oppt_features = ["opptMin", "opptFGM", "opptFGA", "opptFG", "oppt3PM", "oppt3PA", "oppt3PCT", "opptFTM", "opptFTA",
                     "opptFTC", "opptORB", "opptDRB", "opptREB", "opptAST", "opptSTL"
        , "opptBLK", "opptTO", "opptPF", "oppt2P", "opptTS", "opptEFG", "opptPPS", "opptFIC", "opptFIC40", "opptOrtg",
                     "opptDrtg", "opptPlay"]

    oppt_features = oppt_features + feature_names_of_players * 11

    last_features = ["poss", "LM_totalPoint","LM_dayOffset","pace"]

    feature_names_of_matchups = team_features + oppt_features + last_features
    return feature_names_of_matchups
def get_unique_team_abbr(df):
    """

    :param df: Data.csv
    :return: Team Abbrs
    """
    return np.unique(df.teamAbbr.values)
def get_home_team_stats(df,last_x_match=10):
    """

    :param df: Data.csv
    :param last_x_match: Son kaç maç
    :return: Takım sayısı kadar evde yapılan maçların istatistikleri
    """
    team_abbrs = get_unique_team_abbr(df)
    team_stats = pd.DataFrame(columns=df.columns[:41], index=range(len(team_abbrs)))
    team_stats.drop(columns=["opptAbbr"], inplace=True)
    i = 0
    for each in team_abbrs:
        df_for_each = df[df.teamAbbr == each][-1:-last_x_match:-1]
        team_stats_result_count = df_for_each["rslt"].value_counts()
        if team_stats_result_count.shape[0] == 1:
            team_stats_result_count = team_stats_result_count[0] / (
                team_stats_result_count[0]) * 100
        else:
            team_stats_result_count = team_stats_result_count[0] / (
                    team_stats_result_count[0] + team_stats_result_count[1]) * 100

        team_stats.iloc[i, 1] = each
        team_stats.at[i, "rslt"] = team_stats_result_count
        t = df_for_each.iloc[:, :41]
        mean_ = t.iloc[:, 4:].mean()
        team_stats.iloc[i, 3:] = mean_
        i += 1
    team_stats.drop(columns=["gameId"], inplace=True)
    try :
        team_stats.to_csv("D:/NBA-Predict/input/all_home_team_stats.csv")
        print("CSV Saved To Input Folder name all_home_team_stats.csv")
    except Exception as e:
        print(e)
    return team_stats
def get_away_team_stats(df,last_x_match=10):
    """

    :param df: Data.csv
    :param last_x_match: Son kaç maç
    :return: Takım sayısı kadar evde yapılan maçların istatistikleri
    """
    team_abbrs = get_unique_team_abbr(df)
    team_stats = pd.DataFrame(columns=df.columns[:299], index=range(len(team_abbrs)))
    team_stats.drop(columns=["teamAbbr"], inplace=True)
    team_stats.drop(columns=df.columns[4:271],inplace=True)

    i = 0
    for each in team_abbrs:
        df_for_each = df[df.opptAbbr == each][-1:-last_x_match:-1]
        team_stats_result_count = df_for_each["rslt"].value_counts()
        if team_stats_result_count.shape[0] == 1:
            team_stats_result_count = team_stats_result_count[0] / (
                team_stats_result_count[0]) * 100
        else:
            team_stats_result_count = team_stats_result_count[0] / (
                    team_stats_result_count[0] + team_stats_result_count[1]) * 100

        team_stats.iloc[i, 1] = each
        team_stats.at[i, "rslt"] = team_stats_result_count
        cols = team_stats.columns
        t = df_for_each.loc[:,cols]
        print(team_stats.columns)
        mean_ = t.iloc[:, 3:].mean()
        team_stats.iloc[i, 3:] = mean_
        i += 1
    team_stats.drop(columns=["gameId"], inplace=True)
    try :
        team_stats.to_csv("D:/NBA-Predict/input/all_away_team_stats.csv")
        print("CSV Saved To Input Folder name all_home_team_stats.csv")
    except Exception as e:
        print(e)
    return team_stats
def get_unique_player_list(df):
    """
    :param df: Data.csv
    :return: Player List
    """
    players = []
    for i in range(df.PLAYER_NAME.shape[0]):
        for j in range(df.PLAYER_NAME.shape[1]):
            players.append(df.PLAYER_NAME.iloc[i, j])

    unique_players = np.unique(players)
    return unique_players
def create_empty_player_stats(player_list):
    """

    :param player_list: Oyuncu Listesi
    :return: Oyuncu sayısı kadar row olan stats tablosu döndürür
    """
    col_names_for_player_stats = ["PLAYER_NAME", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
                                  "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS"]
    index_for_player_stats = range(len(player_list))
    player_stats_df = pd.DataFrame(columns=col_names_for_player_stats, index=index_for_player_stats)
    return player_stats_df
def get_all_players_stats_df(df):
    """

    :param df: Data.csv --> Dataframe
    :return:Bütün oyuncuların tüm maçlardaki verilerini döndürür
    """
    team_player_first_column_index = df.columns.get_loc("teamPlay") + 1
    team_player_last_column_index = df.columns.get_loc("opptMin")
    oppt_player_first_column_index = df.columns.get_loc("opptPlay") + 1
    oppt_player_last_column_index = df.columns.get_loc("poss")
    team_player_stats = df.iloc[:, team_player_first_column_index:team_player_last_column_index]
    oppt_player_stats = df.iloc[:, oppt_player_first_column_index:oppt_player_last_column_index]
    player_stats_conc = pd.concat([team_player_stats, oppt_player_stats], axis=0)
    player_stats_conc_splited = np.array_split(player_stats_conc, 11, axis=1)
    full_df = pd.concat([player_stats_conc_splited[0],
                         player_stats_conc_splited[1],
                         player_stats_conc_splited[2],
                         player_stats_conc_splited[3],
                         player_stats_conc_splited[4],
                         player_stats_conc_splited[5],
                         player_stats_conc_splited[6],
                         player_stats_conc_splited[7],
                         player_stats_conc_splited[8],
                         player_stats_conc_splited[9],
                         player_stats_conc_splited[10]], axis=0)
    return full_df
def get_all_player_stats_last_x_match(all_player_stats_df,all_player_stats_empty_df,player_list,last_x_match=10):
    """

    :param all_player_stats_df: Maçlara göre olan bütün oyuncuların verileri
    :param all_player_stats_empty_df: Player sayısı kadar row olan boş veri tablosu
    :param player_list: Playerların listesi
    :param last_x_match: Son kaç maç olduğu
    :return: Bütün oyuncuların ortalama verileri
    """
    for i in range(len(player_list)):
        each = player_list[i]
        player_stats_each = all_player_stats_df[all_player_stats_df.PLAYER_NAME == each][-1:-last_x_match:-1]
        player_name = each
        mean_of_stats_this_player = player_stats_each.drop(columns=["PLAYER_NAME"], axis=1).mean()
        all_player_stats_empty_df.at[i, "PLAYER_NAME"] = player_name
        all_player_stats_empty_df.iloc[i, 1:] = mean_of_stats_this_player
    try :
        all_player_stats_empty_df.to_csv("D:/NBA-Predict/input/all_player_stats.csv")
        print("CSV Saved To Input Folder name all_player_stats.csv")
    except Exception as e:
        print(e)
    return all_player_stats_empty_df
def get_player_stats_by_name(all_player_stats_df,player_name,last_x_match=10):
    """

    :param all_player_stats_df: Bütün playerların maçlara göre verileri
    :param player_name: Player ismi
    :param last_x_match: Son kaç maç olduğu
    :return: Playerın ortalama değerleri
    """

    player_stats_each = all_player_stats_df[all_player_stats_df.PLAYER_NAME == player_name][-1:-last_x_match:-1]
    mean_of_stats_this_player = player_stats_each.drop(columns=["PLAYER_NAME"], axis=1).mean()
    # all_player_stats_empty_df.at[i, "PLAYER_NAME"] = player_name
    # all_player_stats_empty_df.iloc[i, 1:] = mean_of_stats_this_player

    return player_name,mean_of_stats_this_player
def get_play_by_play_stats(df,last_x_match=10):
    """

    :param df: Data.csv
    :param last_x_match: Son kaç maç
    :return: Takımların diğer takımlarla olan maç istatistikleri
    """

    """preparing dropped columns"""

    teamAbbr_unique = get_unique_team_abbr(df)
    opptAbbr_unique = get_unique_team_abbr(df)
    coll = df.drop(df.columns[42:272],axis=1)
    coll = df.drop(df.columns[300:530],axis=1)
    coll = coll.drop("gameId",axis=1)
    df = coll

    play_by_play = pd.DataFrame(columns=df.columns, index=range(len(teamAbbr_unique) * len(teamAbbr_unique)))
    i = 0
    for each in teamAbbr_unique:
        for k in opptAbbr_unique:
            if each != k:
                if df[(df.teamAbbr == each) & (df.opptAbbr == k)].empty == False:
                    df_for_each = df[(df.teamAbbr == each) & (df.opptAbbr == k)][-1:-last_x_match:-1]

                    play_by_play_count = df_for_each["rslt"].value_counts()
                    # print(each,k)
                    # print(df[(df.teamAbbr == each) & (df.opptAbbr == k)])
                    if play_by_play_count.shape[0] == 1:
                        play_by_play_count = play_by_play_count[0] / (play_by_play_count[0]) * 100
                    else:
                        play_by_play_count = play_by_play_count[0] / (play_by_play_count[0] + play_by_play_count[1]) * 100
                    play_by_play.iloc[i, 3:] = df[(df.teamAbbr == each) & (df.opptAbbr == k)].iloc[:, 3:].mean()
                    play_by_play.loc[i, "teamAbbr"] = each
                    play_by_play.loc[i, "opptAbbr"] = k
                    play_by_play.at[i, "rslt"] = play_by_play_count

                    i += 1
                else:
                    pass
    try :
        play_by_play.to_csv("D:/NBA-Predict/input/play_by_play.csv")
        print("CSV Saved To Input Folder name play_by_play.csv")
    except Exception as e:
        print(e)
    return play_by_play
def get_test_data(teamAbbr,opptAbbr,homeplayers,awayplayers,data_csv_name = "D:/NBA-Predict/matches_w_player_stats.csv",player_stats_csv_name = "D:/NBA-Predict/input/all_player_stats.csv",last_x_match=10):
    df = pd.read_csv(data_csv_name)
    df.columns = remove_whitespaces_in_df_columns(df)  # clean column names
    df.columns = get_column_names()  # get column names and assign
    all_players_stats_df = pd.read_csv(player_stats_csv_name,index_col=0)
    play_by_play = get_play_by_play_stats(df,last_x_match)
    test_data = play_by_play[(play_by_play["teamAbbr"] == teamAbbr) & (play_by_play["opptAbbr"] == opptAbbr)]
    home = []
    away = []
    feature_names_of_players = ["MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
                                "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS"]
    for each in homeplayers:
        player_name , mean_of_player = get_player_stats_by_name(all_player_stats_df=all_players_stats_df,player_name=each,last_x_match=last_x_match)
        mean_of_player = mean_of_player.tolist()
        home.append(mean_of_player)
    for each in awayplayers:
        player_name , mean_of_player = get_player_stats_by_name(all_player_stats_df=all_players_stats_df,player_name=each,last_x_match=last_x_match)
        mean_of_player = mean_of_player.tolist()
        away.append(mean_of_player)
    all_df = pd.DataFrame(columns=range(len(feature_names_of_players)*(len(homeplayers)+len(awayplayers))),index=range(1))
    start_offset = 0
    end_offset = len(feature_names_of_players)
    for i in range(len(homeplayers)):
        all_df.iloc[0,start_offset:end_offset] = home[i]
        start_offset = end_offset
        end_offset += 20
    start_offset = len(homeplayers) * len(feature_names_of_players)
    end_offset = start_offset + len(feature_names_of_players)
    for i in range(len(awayplayers)):
        all_df.iloc[0,start_offset:end_offset] = away[i]
        start_offset = end_offset
        end_offset += len(feature_names_of_players)
    all_df.columns = feature_names_of_players * (len(homeplayers) + len(awayplayers))
    test_data.reset_index(inplace=True)

    conc =  pd.concat([test_data,all_df],axis=1)
    conc.to_csv("D:/NBA-Predict/test_data/"+teamAbbr+"vs"+opptAbbr+"_test_data.csv")
    conc.fillna(value=0,inplace=True)
    conc.drop("index",axis=1,inplace=True)
    return conc
def drop_some_columns(df):
    df = df.drop(columns=["PLAYER_NAME"],axis=1)
    df = df.drop(columns=["gameId"],axis=1)
    return df
def label_train_data(df):
    le = LabelEncoder()
    df["teamAbbr"] = le.fit_transform(df["teamAbbr"])
    keys = le.classes_
    values = le.transform(le.classes_)
    dictionary1 = dict(zip(keys, values))
    np.save('D:/NBA-Predict/labelencoder/team_abbr.npy', le.classes_)
    df["opptAbbr"] = le.fit_transform(df["opptAbbr"])
    keys = le.classes_
    values = le.transform(le.classes_)
    np.save('D:/NBA-Predict/labelencoder/oppt_abbr.npy', le.classes_)
    dictionary2 = dict(zip(keys, values))
    df["rslt"] = le.fit_transform(df["rslt"])
    keys = le.classes_
    values = le.transform(le.classes_)
    np.save('D:/NBA-Predict/labelencoder/rslt.npy', le.classes_)
    dictionary3 = dict(zip(keys, values))
    return df,dictionary1,dictionary2,dictionary3
def label_test_data(df):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('NBA-Predict/labelencoder/team_abbr.npy',allow_pickle=True)
    df.teamAbbr = encoder.transform(df.teamAbbr)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('D:/NBA-Predict/labelencoder/oppt_abbr.npy',allow_pickle=True)
    df.opptAbbr = encoder.transform(df.opptAbbr)

    return df
def clean_nan_values(df):
    df.dropna(inplace=True)
    return df
def standart_scaler_all_data(df):
    scaler_filename = "D:/NBA-Predict/standartScaler/scaler.bin"
    ss = StandardScaler()
    df.iloc[:, 3:] = ss.fit_transform(df.iloc[:, 3:])
    joblib.dump(ss, scaler_filename,compress=True)
    print("Saved Standart Scaler File to ",scaler_filename)
    return df,ss
def standart_scaler_test_data(df):
    scaler_filename = "D:/NBA-Predict/standartScaler/scaler.bin"
    scaler = joblib.load(scaler_filename)
    df.iloc[:, 3:] = scaler.transform(df.iloc[:, 3:])
    return df
def onehotencoder_all_data(df):
    ohe = OneHotEncoder(categorical_features=[0, 1], n_values='auto',
                        handle_unknown='ignore')

    df = ohe.fit_transform(df).toarray()
    df = pd.DataFrame(df)
    onehotencoder_name = "D:/NBA-Predict/onehotencoder/onehotencoder.bin"
    joblib.dump(ohe, onehotencoder_name, compress=True)
    print("Saved ohe to ",onehotencoder_name)
    return df
def onehotencoder_test_data(df):
    onehotencoder_name = "D:/NBA-Predict/onehotencoder/onehotencoder.bin"
    ohe = joblib.load(onehotencoder_name)
    df = ohe.transform(df).toarray()
    return pd.DataFrame(df)
def old_to_new_team_abbrs(df):
    currently_available_teams = {}
    currently_available_teams["NJN"] = "BKN" # old to new
    currently_available_teams["NOH"] = "NOP" # old to new

    non_teams = ['EST', 'FLA', 'GNS', 'GUA', 'MAC']
    for key, value in currently_available_teams.items():
        df['teamAbbr'] = df['teamAbbr'].str.replace(key, value)
        df["opptAbbr"] = df["opptAbbr"].str.replace(key, value)
    for each in non_teams:
        drop_index = df[(df.teamAbbr == each) | (df.opptAbbr == each)].index
        df.drop(drop_index, inplace=True)
    print(get_unique_team_abbr(df))
    return df
def get_one_shot_player_stats(players,df,last_x_match):
    empty_stats_df = create_empty_player_stats(players)  # create empty stats of teams table
    all_players_stats_df = get_all_players_stats_df(df)  # assign all team stats to empty table
    all_players_stats_last_x_match = get_all_player_stats_last_x_match(all_player_stats_df=all_players_stats_df, all_player_stats_empty_df=empty_stats_df,player_list= players,last_x_match=last_x_match)
def get_available_player_stats():
    csv_url="D:/NBA-Predict/matches_w_player_stats.csv"
    df = pd.read_csv(csv_url,header=None)
    df.columns = get_column_names()
    play_by_play = get_play_by_play_stats(df,last_x_match=1000)
    players = pd.read_csv("D:/NBA-Predict/input/players/player_list.csv",header=None)
    players = players.values
    players = players[:,0]
    get_one_shot_player_stats(players,df)
def current_player_stats(last_x_match):
    csv_url="D:/NBA-Predict/matches_w_player_stats.csv"
    df = pd.read_csv(csv_url, header=None)
    df.columns = get_column_names()
    players = pd.read_csv("D:/NBA-Predict/input/players/player_list.csv", header=None)
    players = players.values
    players = players[:, 0]
    get_one_shot_player_stats(players, df,last_x_match=last_x_match)
