#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#Carga de archivo csv en un dataframe
file = 'AirQuality.csv'
dataset = pd.read_csv(file, sep=';')
dataset.head()
dataset.info()

#Eliminación de las columnas CO(GT) y sin nombre
dataset.drop(['CO(GT)','Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)

#Formateando algunas columnas de objetos de cadenas a flotantes
dataset.replace(to_replace=',',value='.',regex=True,inplace=True)
for i in 'C6H6(GT) T RH AH'.split():
    dataset[i] = pd.to_numeric(dataset[i],errors='coerce')
    
#Reemplazando los datos nulos de -200 a NaN para el tratamiento posterior
dataset.replace(to_replace=-200,value=np.nan,inplace=True)
dataset.info()

#Formateando Date y Time al tipo Datetime
dataset['Date'] = pd.to_datetime(dataset['Date'],dayfirst=True) 
dataset['Time'] = pd.to_datetime(dataset['Time'],format= '%H.%M.%S' ).dt.time
dataset.head()

NMHC_ratio = dataset['NMHC(GT)'].isna().sum()/len(dataset['NMHC(GT)'])
print('El sensor NMHC(GT) tiene un {:.2f}% de datos perdidos.'.format(NMHC_ratio*100))

#Se elimina el sensor NMHC(GT) debido a la cantidad de valores nulos
dataset.drop('NMHC(GT)', axis=1, inplace=True) 
dataset.info()

sns.set_theme(style="whitegrid")
for i in dataset.columns[2:13]:
    sns.boxplot(x=dataset[i])
    plt.title('Diagrama de caja de los datos de los sensores')
    plt.show()

#Eliminación de valores atípicos con el método del rango intercuartil (IQR)
Q1 = dataset.quantile(0.25) #Primer 25% de los datos
Q3 = dataset.quantile(0.75) #Primer 75% de los datos
IQR = Q3 - Q1 #IQR = Rango Intercuartil (InterQuartile Range)
scale = 2
lower_lim = Q1 - scale*IQR
upper_lim = Q3 + scale*IQR
lower_outliers = (dataset[dataset.columns[2:13]] < lower_lim)
upper_outliers = (dataset[dataset.columns[2:13]] > upper_lim)

#Comprobando los valores atípicos resultantes calculados por el método anterior
#(representados a continuación como valores no nulos)
dataset[dataset.columns[2:13]][(lower_outliers | upper_outliers)].info()

#Creación de un nuevo DataFrame sin los valores atípicos
num_cols = list(dataset.columns[2:13])
dataset_out_IQR = dataset[~((dataset[num_cols] < (Q1 - scale * IQR)) |(dataset[num_cols] > (Q3 + scale * IQR))).any(axis=1)]
dataset_out_IQR.info()

#Se eliminan los sensores NOx(GT) y NO2(GT) debido a la cantidad de valores nulos respecto a otros sensores
pd.options.mode.chained_assignment = None
dataset_out_IQR.drop(['NOx(GT)','NO2(GT)'],axis=1, inplace=True)
dataset_out_IQR.info()

#Eliminación de filas con valores NaN 
dataset_filt = dataset_out_IQR.dropna(how='any', axis=0)
dataset_filt.reset_index(drop=True,inplace=True)
dataset_filt.info()

#Agregando una columna con los días de la semana 
dataset_filt['Week Day'] = dataset_filt['Date'].dt.day_name()

#Reorganización de las columnas
cols = dataset_filt.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:11]
dataset_filt = dataset_filt[cols]
dataset_filt.head(10)

#Creación de un nuevo Dataframe con los datos del miércoles
dataset_wed = dataset_filt[dataset_filt['Week Day'] == 'Wednesday']

#Planificación del valor medio horario de CO los miércoles

sns.barplot(x='Time',y='PT08.S1(CO)', data=dataset_wed.sort_values('Time'))
plt.title('Valores promedio por hora del CO los miércoles')
plt.xticks(rotation=90)
plt.show()

#Los picos de concentración de CO se dan entre 8 y 9 AM y entre 6 y 8 PM.
#Al principio y al final del horario de oficina, respectivamente.
#¿Coincidencia? No lo creo :v

#Graficando la matriz de correlación
sns.heatmap(dataset.corr(),annot=True,cmap = 'coolwarm')
plt.title('Matriz de correlación')
plt.show()

sns.pairplot(dataset)
plt.show()

#Eliminando el C6H6(GT) por ser un sensor redundante
#La molécula C6HC es un hidrocarburo no metánico (NMHC)
#Por consiguiente, la correlación entre estos dos sensores es exacta (o casi)
dataset_filt.drop('C6H6(GT)', axis=1, inplace=True)

#Creación de un modelo de regresión del sensor PT08.S1(CO)
Y = dataset_filt['PT08.S1(CO)'] #variável de predição
X = dataset_filt.drop(['PT08.S1(CO)','Date', 'Time', 'Week Day'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

modelo = LinearRegression()
modelo.fit(X = np.array(X_train), y = Y_train)
pred_modelo = modelo.predict(X_test)#Concentraciones de CO pronosticadas

#Evaluando los resultados con la métrica R²:
#Evaluación de datos de prueba

print('Modelo de regresión de bosque aleatorio: R²={:.2f}'.format(metrics.r2_score(Y_test, pred_modelo)))

#Comparación de las predicciones del modelo con los datos reales
aux = pd.DataFrame()
aux['Y_test'] = Y_test
aux['Pronósticos Modelo_01'] = pred_modelo
plt.figure(figsize=(15,5))
sns.lineplot(data=aux.iloc[:200,:])
plt.show()

#Extracción de la información de la estación de la columna Date:
#https://ciencia.nasa.gov/disminucion-global-del-monoxido-de-carbono#:~:text=La%20reducci%C3%B3n%20de%20la%20luz,invierno%20y%20principios%20de%20primavera.
def season(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'

dataset_filt['Season'] = dataset_filt['Date'].map(season)
dataset_filt.head(10)

sns.pairplot(dataset_filt, hue='Season')
plt.show()

#Creando características categóricas a partir de la columna Season y dividiendo el nuevo Dataframe
Y_2 = dataset_filt['PT08.S1(CO)']
X_2 = dataset_filt.drop(['PT08.S1(CO)','Date', 'Time', 'Week Day'], axis=1)
#https://interactivechaos.com/es/manual/tutorial-de-machine-learning/la-funcion-getdummies
X_2 = pd.get_dummies(data=X_2)

X_2_train, X_2_test, Y_2_train, Y_2_test = train_test_split(X_2, Y_2, test_size=0.2, random_state=42)
X_2.head()

#Nuevo modelo con estaciones
modelo_2 = LinearRegression()
modelo_2.fit(X_2_train, Y_2_train)

pred_modelo_2 = modelo_2.predict(X_2_test)

print('Modelo de regresión sin estaciones: R²={:.2f}'.format(metrics.r2_score(Y_test, pred_modelo)))
print('Modelo de regresión con estaciones: R²={:.2f}'.format(metrics.r2_score(Y_2_test, pred_modelo_2)))

#Comparación de las predicciones del nuevo modelo con los datos reales y el modelo sin estaciones
aux['Pronósticos Modelo_02'] = pred_modelo_2
plt.figure(figsize=(15,5))
sns.lineplot(data=aux.iloc[:200,:])
plt.show()