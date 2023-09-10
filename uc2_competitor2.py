# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:26:28 2023

@author: BEJ02
"""


import pandas as pd
from datetime import datetime
import pymssql
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 25)
pd.set_option('display.min_rows', 50)
pd.set_option('display.max_colwidth', 70)



#%% data ingestion

def get_table_from_sql(query_string):
    server ='dbbi1.simplex.ca'
    database = 'PORDB_POR'
    username = 'jfbernier'
    password = 'Simplex1907*'
    conn = pymssql.connect( server,username, password, database)
    sqltable = pd.read_sql(query_string, conn)
    conn.close()
    return sqltable

# pickup_day_of_week
# pickup_month
# cor_day_of_week
# cor_month
# type of truck (LVP or else)
# product super_category
# transaction sous_region

def get_training_data():

    sql1 = """
        select 
        mrd.contractNumber as CLE_TRANSACTION
        ,e.SOUS_REGION AS SOUS_REGION
        ,t.TOTL as TOTAL_TRANSACTION
        ,ti.ITEM AS CODE_PRODUIT
        ,ra.DATE_APPEL as DATE_APPEL_FIN_LOCATION
        ,t.PickupDate AS DATE_RAMASSAGE
        
        FROM [PORDB_POR].[dbo].[MapRouteDetails] as mrd
        left join (SELECT CNTR, case when LEFT(CONVERT(VARCHAR,CNTR), 1) = 'L' then JBPO else CNTR END trans_link 
        			FROM Transactions) as tl 
        			on mrd.contractNumber = tl.CNTR
        
        left join (select CNTR, ITEM, DDT 
        			from qryTransactionItems WHERE upper(left(txty, 1)) = 'R')as ti 
        			on tl.trans_link  = ti.CNTR 
        
        left join Transactions as t on tl.trans_link = t.CNTR
        
        left join (select contrat_liee, date_appel 
        		    from vg_registre_appels where CUEILLETTE = 1) as ra 
        			on tl.trans_link  = ra.CONTRAT_LIEE and mrd.destinationType = '2'
        
        left join t_emplacement as e on mrd.contractStore  = e.cle_emplacement
        
        WHERE mrd.tripStatusString = 'Pick Up' and ra.DATE_APPEL > dateadd(day, -730, GETDATE())
    
    """
    
    counter = 0
    while counter < 3 :
        print("\n try #"+ str(counter))
        try :
            df = get_table_from_sql(sql1)
            print(df.head())
            counter = 99
            
        except  Exception as e:
            time.sleep(5)
            counter = counter + 1 
            print(e)
    
    #print("import done : "+str(len(df))+ " lines imported")

    df["DATE_APPEL_FIN_LOCATION"] = pd.to_datetime(df['DATE_APPEL_FIN_LOCATION'])
    df['fin_location_hour'] = df['DATE_APPEL_FIN_LOCATION'].dt.hour
    df['fin_location_dow'] = df['DATE_APPEL_FIN_LOCATION'].dt.dayofweek
    df['fin_location_day'] = df['DATE_APPEL_FIN_LOCATION'].dt.day
    df['fin_location_mois'] = df['DATE_APPEL_FIN_LOCATION'].dt.month
    
    df["DATE_RAMASSAGE"] = pd.to_datetime(df['DATE_RAMASSAGE'])
    #df['ramassage_dow'] = df['DATE_RAMASSAGE'].dt.dayofweek
    #df['ramassage_mois'] = df['DATE_RAMASSAGE'].dt.month
    
    df['duration'] = (df["DATE_RAMASSAGE"]-df["DATE_APPEL_FIN_LOCATION"]).dt.days
    
    training_df = df[df['duration'] >= 0]
    
    print("preprocessing done")
    
    return training_df
    
def remap_classes(d):
    if d == 0:
        return 0
    if d == 1:
        return 1
    if d >= 2:
        return 2
    else :
        return -1
    

def train_model(df, target, explainatory, show_importance = True):
    # define X
    X = df[explainatory]
    X = pd.get_dummies(X).astype(float)
    train_features = list(X.columns)

    df['y_cat'] = df[target].apply(remap_classes)
    y_cat = df['y_cat'].astype(float)

    modelc = RandomForestClassifier(n_estimators= 100, min_samples_leaf = 3,  max_depth=15, random_state=0)

    modelc.fit(X, y_cat)
    
    if show_importance:
        print('classification importance')
        l1 = list(X.columns)
        l2 = list(modelc.feature_importances_)
        d = {"feature":l1, "importance":l2}
        imp = pd.DataFrame(d)
        imp = imp.sort_values(by = "importance", ascending=False)
        print(imp.head(10))

    df['y_classification'] = modelc.predict(X)
    yhat = modelc.predict_proba(X)

    df['prob_0'] = yhat[:,0]
    df['prob_1'] = yhat[:,1]
    df['prob_2'] = yhat[:,2]
    

    def check_valid_prediction(row):
        # prédiction est valide si la prédiction est plus grande que le baseline
        if int(row['y_classification']) == int(row['y_cat']):
            return 1
        
        valid = []
        if row['prob_0'] > 0.333 :
            valid.append(0)
        elif row['prob_1'] > 0.333 :
            valid.append(1)
        elif row['prob_2'] > 0.333 :
            valid.append(2)
                    
        if int(row['y_classification']) in valid:
            return 1
        else :
            return 0
        
    df['valid_prediction'] = df.apply(check_valid_prediction, axis = 1)
    
    #print(df['valid_prediction'].sum()/len(df['valid_prediction']))

    return modelc, train_features


def get_ready_for_pickups():
    
    sql1 = """
            select
                t.cntr as CLE_TRANSACTION
            	,ti.ITEM AS CODE_PRODUIT
                ,e.SOUS_REGION AS SOUS_REGION
                ,t.TOTL as TOTAL_TRANSACTION
                ,ra.DATE_APPEL as DATE_APPEL_FIN_LOCATION
            
            
            from transactions as t
            left join (SELECT CNTR, case when LEFT(CONVERT(VARCHAR,CNTR), 1) = 'L' then JBPO else CNTR END trans_link 
                			FROM Transactions) as tl 
                			on t.CNTR = tl.CNTR
            left join TransactionItems as ti on tl.cntr = ti.cntr
            left join t_emplacement as e on t.str = e.cle_emplacement
            left join (select contrat_liee, date_appel 
                		    from vg_registre_appels where CUEILLETTE = 1) as ra 
                			on tl.trans_link  = ra.CONTRAT_LIEE
            
            where upper(ti.txty)= 'RH' AND upper(left(CONVERT(VARCHAR, t.stat), 1))= 'O'

    """

    df = get_table_from_sql(sql1)
    df["DATE_APPEL_FIN_LOCATION"] = pd.to_datetime(df['DATE_APPEL_FIN_LOCATION'])
    df['fin_location_hour'] = df['DATE_APPEL_FIN_LOCATION'].dt.hour
    df['fin_location_dow'] = df['DATE_APPEL_FIN_LOCATION'].dt.dayofweek
    df['fin_location_day'] = df['DATE_APPEL_FIN_LOCATION'].dt.day
    df['fin_location_mois'] = df['DATE_APPEL_FIN_LOCATION'].dt.month
    
    return df

def predict_data(model, train_features, df, explainatory):
    
    df.dropna(inplace = True)
    X = df[explainatory]
    X = pd.get_dummies(X).astype(float)
    for f in train_features:
        if f not in X.columns:
            X[f] = 0.0
    X = X[train_features]
    
    
    df['y_hat'] = model.predict(X)
    yhat = model.predict_proba(X)

    df['prob_0'] = yhat[:,0]
    df['prob_1'] = yhat[:,1]
    df['prob_2'] = yhat[:,2]
    
    df = df[["CLE_TRANSACTION", "CODE_PRODUIT", "prob_0", "prob_1", "prob_2"]]
    df.drop_duplicates(inplace = True)
    
    
    # tag prediction with a string 
    def tag_prediction(row):
        valid = []
        baseline = 0.333
        if row['prob_0'] > baseline :
            valid.append(0)
        if row['prob_1'] > baseline :
            valid.append(1)
        if row['prob_2'] > baseline :
            valid.append(2)
            
        return ", ".join((str(element) for element in valid))
    df['PROB_0'] = df['prob_0'].round(2)
    df['PROB_1'] = df['prob_1'].round(2)
    df['PROB_2'] = df['prob_2'].round(2)
    df['PREDICTION'] = df.apply(tag_prediction, axis = 1)
    df['DATE_IMPORT'] = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df = df[["CLE_TRANSACTION", "CODE_PRODUIT", "PREDICTION", "DATE_IMPORT", "PROB_0", "PROB_1", "PROB_2"]]
    return df


def push_data(df_sql, reset = False):
    server = "dbbi1.simplex.ca"
    database = "PORDB_POR"
    username = "jfbernier"
    password = "Simplex1907*"
    conn = pymssql.connect( server,username, password, database)
    cur = conn.cursor()
    print("connexion created")
    
    if reset :
        cur.execute("""IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[t_uc2_jf]') AND type in (N'U'))
                        DROP TABLE [dbo].[t_uc2_jf]""")
        conn.commit()
        print("table dropped")
        
        cur.execute(""" USE PORDB_POR;
    
                    CREATE TABLE t_uc2_jf (
                      CLE_TRANSACTION nvarchar(30),
                      CODE_PRODUIT nvarchar(30),
                      PREDICTION nvarchar(10),
                      DATE_IMPORT nvarchar(30),
                      PROB_0 float,
                      PROB_1 float,
                      PROB_2 float
                    )
                    """)
        conn.commit()
        print("new table created")
    

    query = """INSERT INTO dbo.t_uc2_jf (CLE_TRANSACTION, CODE_PRODUIT, PREDICTION, DATE_IMPORT, PROB_0, PROB_1, PROB_2)
                VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    
    sql_data = tuple(map(tuple, df_sql.values))
    cur.executemany(query, sql_data)
    print("data uploaded")
    
    
    conn.commit()
    cur.close()
    conn.close()

def main():
    training_df = get_training_data()
    training_target =  "duration"
    training_features = ['SOUS_REGION', 'TOTAL_TRANSACTION', 'fin_location_hour', "fin_location_dow", "fin_location_day", "fin_location_mois"]
    model, t = train_model(training_df, training_target, training_features, True)
    inference_data = get_ready_for_pickups()
    output_data = predict_data(model, t, inference_data, training_features)
    push_data(output_data)

if __name__ == "__main__":
    main()



