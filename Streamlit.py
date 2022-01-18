#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from sklearn import preprocessing
import pandas as pd
import numpy as np
import pickle
import ast


# In[2]:


train2 = pd.read_csv('train_df2.csv',index_col=False)
train1 = pd.read_csv('train_df1.csv',index_col=False)


# In[3]:


train1['revenue'] = np.log1p(train1['revenue'])
train1['log_budget'] = np.log1p(train1['budget'])
train1['log_popularity'] = np.log1p(train1['popularity'])


train1['ratio_budget_runtime'] = (train1['log_budget'] / train1['runtime'])
train1['ratio_budget_popularity'] = train1['log_budget'] / train1['log_popularity']
train1['ratio_budget_year'] = train1['log_budget'] / train1['year'] 
train1['ratio_popularity_year'] = train1['log_popularity'] / train1['year']
train1['ratio_runtime_year'] = train1['runtime'] / train1['year']
train1['ratio_budget_year2'] = train1['log_budget'] / (train1['year']*train1['year'])
train1['ratio_year_budget'] = train1['year'] / train1['log_budget']
train1['popularity_runtime_to_budget'] = train1['log_popularity'] / train1['ratio_budget_runtime']    
train1['budget_to_runtime_to_year'] = train1['ratio_budget_runtime'] / train1['year']
train1['ratio_year_popularity'] = train1['year'] / train1['log_popularity']

del train1['Unnamed: 0']


# In[4]:


col = ['budget', 'popularity', 'runtime', 'rate',
       'number_genres', 'number_companies', 'number_crew','number_cast', 'numberVotes', 'has_collection',
        'is_eng', 'year', 'month','Top_cast_revenue', 'Top_Director_revenue',
       'Top_Writer_revenue', 'Top_Producer_revenue',
       'Top_Original Music Composer_revenue','Top_companies_revenue','Action', 'Adventure', 'Drama', 'Comedy',
         'Thriller', 'Family',
       'Science Fiction', 'Fantasy', 'Romance', 'Crime', 'weighted_rate',
       'log_budget', 'log_popularity', 'ratio_budget_runtime',
       'ratio_budget_popularity', 'ratio_budget_year', 'ratio_popularity_year',
       'ratio_runtime_year', 'ratio_budget_year2', 'ratio_year_budget',
       'popularity_runtime_to_budget', 'budget_to_runtime_to_year',
       'ratio_year_popularity']


# In[5]:


from sklearn.model_selection import train_test_split
X = train2[col]
y = train2['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=1)


# In[6]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def evaluate(model, X_test, y_test):
    global accuracy
    global rmse
    global mae
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    rmse = np.sqrt(mean_squared_error(predictions, y_test))
    mae = mean_absolute_error (predictions, y_test)

    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('RMSE = %.3f' % rmse) 
    print('MAE = %.3f' % mae)   
    return accuracy


# In[7]:


def stringtolist(var,save): #some columns have data a list but in string mode. the function convert it to a list.
    for i in var.index:
        try:
            x = ast.literal_eval(var[i])
            save[i] = x
        except:
            save[i] = np.nan

def top30by(dataset,var,by):
    global top30
    size = len(dataset) - dataset[var].isna().sum()
    keywords_dict = {}
    for i in dataset[[by,var]].values:
        try:
            for key in i[1]:
                if key not in keywords_dict:
                    keywords_dict[key] = i[0]
                else:
                    keywords_dict[key] += i[0]
        except:
            pass
    for key in keywords_dict:
        keywords_dict[key] = keywords_dict[key] / size
    keywords_df = pd.DataFrame.from_dict(keywords_dict, orient='index', columns=['mean_movies_'+by])
    top30 = keywords_df.sort_values(by = 'mean_movies_'+by ,ascending = False)[:30]


# In[ ]:





# In[8]:


top_cast_name=['Samuel L. Jackson','Robert De Niro','Morgan Freeman','Liam Neeson','Bruce Willis','Susan Sarandon',
'J.K. Simmons','Bruce McGill','John Turturro','Willem Dafoe','Sylvester Stallone','Owen Wilson',
'Bill Murray','Forest Whitaker','John Goodman','Jason Statham','Sigourney Weaver','Frank Welker',
'George Clooney','Nicolas Cage','Mel Gibson','Keith David''Denzel Washington','Michael Caine',
'Matt Damon','Robert Duvall','Richard Jenkins','Dennis Quaid','Jim Broadbent','Mark Wahlberg']

top_companies_name=['Warner Bros.','Universal Pictures','Paramount Pictures',
                    'Twentieth Century Fox Film Corporation',
'Columbia Pictures','Metro-Goldwyn-Mayer (MGM)','New Line Cinema','Walt Disney Pictures','Touchstone Pictures',
'Columbia Pictures Corporation','TriStar Pictures','Relativity Media','United Artists','Canal+','Miramax Films',
'Village Roadshow Pictures','Regency Enterprises','Dune Entertainment','Working Title Films','DreamWorks SKG',
'Fox Searchlight Pictures','Lionsgate','BBC Films','StudioCanal','Fox 2000 Pictures','Amblin Entertainment',
'Summit Entertainment','Hollywood Pictures','Orion Pictures','Original Film']

top_Director_name=['Ron Howard','Steven Spielberg','Clint Eastwood','Woody Allen','Michael Bay',
                   'Paul W.S. Anderson',
 'Blake Edwards','Francis Ford Coppola','Peter Jackson','Brian De Palma','Wes Craven','Steven Soderbergh',
 'Martin Scorsese','Ridley Scott','Alfred Hitchcock','Joel Schumacher','Tim Burton','Michael Mann',
 'Robert Rodriguez','Peter Hyams','Todd Phillips','Ivan Reitman','Wolfgang Petersen','Roger Donaldson',
 'Lasse Hallstr√∂m','Rob Reiner','Garry Marshall','Billy Wilder','Renny Harlin','Walter Hill']

top_Writer_name=['Zak Penn', 'Woody Allen', 'M. Night Shyamalan', 'Allan Loeb', 'Timur Bekmambetov', 'Tyler Perry',
                 'Nia Vardalos', 'Andrew Niccol', 'David S. Goyer', 'Neil Simon', 'John Sayles',
                 'Richard Linklater', 'Brian Helgeland', 'Ben Hecht', 'Terrence Malick', 'Chris Morgan', 
                 'George Miller', 'Paul W.S. Anderson', 'Matt Sazama', 'Burk Sharpless', 'Simon Barrett',
                 'Todd Phillips', 'Neill Blomkamp', 'David Leslie Johnson', 'Quentin Tarantino', 
                 'Christopher B. Landon', 'Wallace Wolodarsky', 'Steve Rudnick', 'Leo Benvenuti',
                 'Karen McCullah Lutz']

top_Producer_name=['Neal H. Moritz', 'Joel Silver', 'Brian Grazer', 'Scott Rudin', 'Eric Fellner', 'Tim Bevan',
                   'Arnon Milchan', 'Roger Birnbaum', 'Jason Blum', 'Jerry Bruckheimer', 'Lauren Shuler Donner',
                   'Luc Besson', 'Ivan Reitman', 'Gary Lucchesi', 'Tom Rosenberg', 'Gary Barber', 
                   'Michael G. Wilson', 'Ron Howard', 'Menahem Golan', 'Lorenzo di Bonaventura', 
                   'Gale Anne Hurd', 'Charles Roven', 'James G. Robinson', 'John Davis', 'Yoram Globus', 
                   'Ridley Scott', 'Peter Jackson', 'Stacey Sher', 'Michael Shamberg', 'Robert De Niro']

top_music_name=['James Newton Howard', 'Jerry Goldsmith', 'James Horner', 'Hans Zimmer', 'Danny Elfman',
                'John Williams', 'John Debney', 'John Powell', 'Graeme Revell', 'Christophe Beck',
                'Alan Silvestri', 'Marco Beltrami', 'Thomas Newman', 'Mark Isham', 'Howard Shore',
                'Carter Burwell', 'Michael Kamen', 'David Newman', 'Brian Tyler', 'Alexandre Desplat',
                'Mark Mothersbaugh', 'John Barry', 'Christopher Young', 'Clint Mansell', 
                'Harry Gregson-Williams', 'Maurice Jarre', 'Trevor Rabin', 'Rolfe Kent', 'Lalo Schifrin',
                'Michael Giacchino']


# In[9]:


print(top_music_name)


# In[10]:


genres=['Action', 'Adventure', 'Drama', 'Comedy','Thriller', 'Family','Science Fiction', 'Fantasy', 'Romance',
       'Crime']

df = pd.DataFrame(index = [0],columns=col)
for i in df.columns:
    df[i][0] = 0
#genres , number_genres
def genre(x):
    df['number_genres']=len(x)
    for gen in df.columns:
        for i in x:
            if gen == i:
                df[gen] = 1
    
# weighted_rate , rate , numberVotes
def weighted(number,rate):
    r = rate
    v = number
    c = train1['rate'].mean()
    m = 25000
    df['weighted_rate']=((v / (v + m)) * r) + ((m / (v + m)) * c)
    df['numberVotes']=number
    df['rate']=rate

months_name=['January','February','March','April','May','June','July','August','September','October',
             'November','December']
def others(budget, popularity, runtime,year, month, collection, English,number_companies,number_crew,
           number_cast,top_Director,top_Writer,top_Producer,top_music,top_companies,top_cast):

    months_number=list(np.arange(1,13))
    m = dict(zip(months_name,months_number))
    
    if collection == True:
        df['has_collection'] == 1
    else:
        df['has_collection'] == 0
    if English == True:
        df['is_eng'] == 1
    else:
        df['is_eng'] == 0
        
    
    df['budget']=budget
    df['log_budget'] = np.log1p(df['budget'])
    df['popularity']=popularity
    df['log_popularity'] = np.log1p(df['popularity'])
    df['runtime']=runtime
    df['year']=year
    df['month']=m[month]
    df['number_companies'] = number_companies
    df['number_crew'] = number_crew
    df['number_cast'] = number_cast
    df['ratio_budget_runtime'] = (df['log_budget'] / df['runtime'])
    df['ratio_budget_popularity'] = df['log_budget'] / df['log_popularity']
    df['ratio_budget_year'] = df['log_budget'] / df['year'] 
    df['ratio_popularity_year'] = df['log_popularity'] / df['year']
    df['ratio_runtime_year'] = df['runtime'] / df['year']
    df['ratio_budget_year2'] = df['log_budget'] / (df['year']*df['year'])
    df['ratio_year_budget'] = df['year'] / df['log_budget']
    df['popularity_runtime_to_budget'] = df['log_popularity'] / df['ratio_budget_runtime']    
    df['budget_to_runtime_to_year'] = df['ratio_budget_runtime'] / df['year']
    df['ratio_year_popularity'] = df['year'] / df['log_popularity']
    
    df['Top_Director_revenue'] = len(top_Director)
    df['Top_Writer_revenue'] = len(top_Writer)
    df['Top_Producer_revenue'] = len(top_Producer)
    df['Top_Original Music Composer_revenue'] = len(top_music)
    df['Top_companies_revenue'] = len(top_companies)
    df['Top_cast_revenue'] = len(top_cast)


# In[11]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def scale(col):
    test = train1[[col]]
    test1 = pd.DataFrame(index = [0],columns = [col])
    test1[col][0] = df[col][0]
    test.append(test1)
    min_max_scaler = preprocessing.MinMaxScaler()
    test[col] = min_max_scaler.fit_transform(pd.DataFrame(test[[col]]))
    om = test.iloc[-1]
    df[col] = om[0]


# In[12]:


st.title('TMDB BOX OFFICE by Omid Ghamiloo')      

genres = st.multiselect(label='Choice of levels for genres', options = genres)
budget = int(st.number_input("budget"))
numberVotes = int(st.number_input("numberVotes") )
runtime = int(st.number_input("runtime") )
popularity = float(st.number_input("popularity") )
rate = float(st.number_input("rate"))
year = st.selectbox(label='Choose a year ', options = list(np.arange(1900,2051)))
month = st.selectbox(label='Choose a year ', options = months_name)
collection = st.checkbox("is in a collection? ")
English = st.checkbox("is English? ")
number_cast = st.selectbox(label='Choose number of cast ', options = list(np.arange(1,50)))
number_companies = st.selectbox(label='Choose number of companies ', options = list(np.arange(1,11)))
number_crew = st.selectbox(label='Choose number of crew ', options = list(np.arange(1,50)))
top_cast = st.multiselect(label='which one of these actors, act in this movie?', options = top_cast_name)
top_companies = st.multiselect(label='which one of these companies, created this movie?', options = top_companies_name)
top_Director = st.multiselect(label='which one of these Directors, directed the movie?', options = top_Director_name)
top_Writer = st.multiselect(label='which one of these writers, wrote this movie?', options = top_Writer_name)
top_Producer = st.multiselect(label='which one of these Prodicers, produced this movie?', options = top_Producer_name)
top_music = st.multiselect(label='which one of these musicion, created the music of this movie?', options = top_music_name)
 


# In[13]:


genre(genres)
weighted(numberVotes,rate)
others(budget, popularity, runtime,year, month, collection, English,number_companies,number_crew,
           number_cast,top_Director,top_Writer,top_Producer,top_music,top_companies,top_cast)


# In[14]:


var_test = ['numberVotes','budget','popularity','runtime','weighted_rate','rate','ratio_budget_popularity',
                    'ratio_year_budget','popularity_runtime_to_budget','ratio_year_popularity']
for i in var_test:
    scale(i)


# In[15]:


df = df.fillna(0)


# In[16]:


pickle_in = open('linear_model.pkl', 'rb') 
linear_model = pickle.load(pickle_in)
pickle_in = open('lasso_model.pkl', 'rb') 
lasso_model = pickle.load(pickle_in)
pickle_in = open('ridge_model.pkl', 'rb') 
ridge_model = pickle.load(pickle_in)
pickle_in = open('elastic_model.pkl', 'rb') 
elastic_model = pickle.load(pickle_in)
pickle_in = open('random_model.pkl', 'rb') 
random_model = pickle.load(pickle_in)
pickle_in = open('lgb_model.pkl', 'rb') 
lgb_model = pickle.load(pickle_in)
pickle_in = open('cat_model.pkl', 'rb') 
cat_model = pickle.load(pickle_in)


# In[17]:


models_text = ['linear_model','lasso_model','ridge_model','elastic_model','random_model','cat_model','lgb_model']
models = [linear_model,lasso_model,ridge_model,elastic_model,random_model,cat_model,lgb_model]

model = st.sidebar.selectbox(label='please choose one of this models', options = models_text)
def prediction(model):
    result = model.predict(df)
    return np.exp(result)


# In[18]:


evali = pd.read_csv('evaluation.csv',index_col=False)
evali['name'] = models_text


# In[20]:


for i,j in zip(models_text,np.arange(0,7)):
        if model == i:
            st.sidebar.write(evali[evali['name']==i]['Unnamed: 0'][j])
            st.sidebar.write('RMSE: ',round(evali[evali['name']==i]['RMSE_tuned'][j],3))
            st.sidebar.write('MAE: ',round(evali[evali['name']==i]['MAE_tuned'][j],3))
            st.sidebar.write('Accuracy: ',round(evali[evali['name']==i]['Accuracy_tuned'][j],3))
            st.sidebar.write('Cross Validation ',round(evali[evali['name']==i]['CV_tuned'][j],3))
            st.sidebar.write('RMSE improvement: ',round(evali[evali['name']==i]['RMSE_Improve'][j],3))
            st.sidebar.write('MAE improvement: ',round(evali[evali['name']==i]['MAE_Improve'][j],3))
            st.sidebar.write('Accuracy improvement: ',round(evali[evali['name']==i]['Accuracy_Improve'][j],3))
            st.sidebar.write('Validation improvement: ',round(evali[evali['name']==i]['CV_Improve'][j],3))


# In[21]:


def select_model(model):
    for i,j in zip(models_text,models):
        if model == i:
            n = prediction(j)
            return n


# In[22]:


if st.button("Predict"): 
    result = select_model(model) 
    st.success(result) 

