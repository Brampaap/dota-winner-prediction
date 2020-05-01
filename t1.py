import numpy as np
import sklearn as sk
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Считываем данные
data_train = pd.read_csv("features.csv")
data_test = pd.read_csv("features_test.csv")
#1. Какие признаки имеют пропуски среди своих значений? 
#Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?

null_columns = data_train.columns[data_train.isnull().any()] #Возвращает столбцы, в которых есть пропущенные значения
print(data_train[null_columns].isnull().sum()) #Ввод количества пропусков в null_columns столбцах
data_train = data_train.fillna(0)
data_test = data_test.fillna(0)
 
#Ответ: Выбранные столбцы с пропусками: first_blood_team, radiant_courier_time. 
#В первом, вероятно, что существую матчи, где ни одна из команд не сделала первое 
#убийство в течение 5-ти минут от начала матча. Во втором - пропуски означают, 
#что сторона света не купила курьера в течение первых 5-ти минут от начала матча.


#2. Как называется столбец, содержащий целевую переменную?
#Ответ: Столбец, содержащий целевую переменную - «radiant_win».

Y_train = data_train["radiant_win"]
X_train = data_train.drop(["radiant_win", "duration", "tower_status_radiant",\
	"tower_status_dire", "barracks_status_radiant", "barracks_status_dire"],\
	 axis=1)
#Для X_train удаляем столбцы, которые отсуствуют в тестовой выборке, а также Y_train


#3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? 
#Какое качество при этом получилось?

#Разбиение выборки

kf = KFold(n_splits=5, shuffle=True)

# #Подход 1:===============
# #Классификатор
steps = [10, 20, 30]
for i in steps:
	gbc = GradientBoostingClassifier(n_estimators = i)

# #Кросс-валидация
	start_time = datetime.datetime.now()
	test = pd.DataFrame(sk.model_selection.cross_val_score(n_jobs=2, X=X_train, y=Y_train, estimator=gbc, cv=kf, scoring='roc_auc')).mean()
	print('Time elapsed:', datetime.datetime.now() - start_time)
	print("Esimators\t", i, "Score\t", test[0])


#Время кросс-валидации
# Time elapsed: 0:00:23.728958
# Esimators	 10 Score	 0.6644953723341765
# Time elapsed: 0:00:44.774965
# Esimators	 20 Score	 0.6818074777714974
#Ответ: _____________НИЖЕ________________
# Time elapsed: 0:01:05.528637
# Esimators	 30 Score	 0.688455088489101

#4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? 
#Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?

#Ответ: Определённо, смысл использовать больше 30 деревьев в градиентном бустанге ЕСТЬ,
#но время на обучение тоже будет расти. Большее число деревьев позволяет покрывать большее число остатков.
#Eсли выборка репрезентативна, тогда такой подход будет эффективным.
#Для увеличения скорости обучения при увеличении количества деревьев можно: 
#1) Использовать подвыборку (теряем качество), 
#2) Задействовать больше вычислительных ресурсов,
#3) Поиграть с параметрами (lineiar_rate, max_depth и другие).

#Поход 2:===============

#1. Какое качество получилось у логистической регрессии над всеми исходными признаками? 
#Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? 
#Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?


#Стандартизация
scaler = StandardScaler()

#Кросс-валидация с помощью класса Grid (Неделя 3)
def CrosValLog():
	start_time = datetime.datetime.now()
	kf = KFold(n_splits=5, shuffle=True)
	grid = {'C':np.power(10.0, np.arange(-5, 6))}
	lgc = LogisticRegression(penalty="l2")
	gs = GridSearchCV(lgc, grid, scoring='roc_auc', cv=kf)
	gs.fit(X_train, Y_train)
	print('Time elapsed:', datetime.datetime.now() - start_time)
	print(gs.best_params_)
	print(gs.best_score_)
	return gs.best_params_['C']


CrosValLog()#{'C': 0.01}   0.7162964186682237  Time elapsed: 0:00:25.873934


#Ответ: Лучшее значение параметра C и качество при нём описаны в строке 103.
#Качество логистической регрессии выше. Разница в качестве объясняется тем, что квадратическая функция потерь в
#градиентном бустинге накладывает слишком низкие штрафы.
#Логистическая регрессия работает быстрее.

#2.Как влияет на качество логистической регрессии удаление категориальных признаков 
#(укажите новое значение метрики качества)? 
#Чем вы можете объяснить это изменение?

#Удаление категориальных признаков
X_train = X_train.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero',\
	"d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], axis=1)

CrosValLog() #{'C': 0.01} 0.7164823873552106 Time elapsed: 0:00:25.645773

#Ответ: При удалении категориальных признаков качество увеличивается совсем немного,
#причиной тому является стандартизация признаков, мы привели категории в сравнимый вид,
#что некорректно. Добавляя их в выборку, мы путаем классификатор, что отображается на качестве
#его работы. (Новые метрики качества записаны в строке 119)


#3. Сколько различных идентификаторов героев существует в данной игре?
#Ответ: 112


#4. Какое получилось качество при добавлении "мешка слов" по героям? 
#Улучшилось ли оно по сравнению с предыдущим вариантом? 
#Чем вы можете это объяснить?

def find_X_train(data_train, X_train):
	#Подсчёт числа уникальных значений и выбор максимального ID
	un = pd.unique(data_train[['r1_hero','r2_hero','r3_hero','r4_hero','r5_hero',\
	"d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"]].values.reshape(-1))
	print(np.max(un))


	#Добавление мешка слов к выборке и стандартизация
	X_pick = np.zeros((data_train.shape[0], np.max(un)))
	for i, match_id in enumerate(data_train["match_id"]):
		for p in range(5):
			X_pick[i, data_train.loc[i, 'r{}_hero'.format(p+1)]-1] = 1
			X_pick[i, data_train.loc[i, 'd{}_hero'.format(p+1)]-1] = -1
	X_train = X_train.join(pd.DataFrame(X_pick))
	X_train = pd.DataFrame(scaler.fit_transform(X_train))
	return X_train

X_train = find_X_train(data_train, X_train)
c = CrosValLog() #{'C': 0.01} 0.7517947196144443 Time elapsed: 0:00:40.114561

#Ответ: Наблюдается повышение качества, т.к. в классификации теперь учитываются различные наборы
#пиков для команд, что весомо влияет на исход игры. Метрики описаны в строке 153

#5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось
#у лучшего из алгоритмов?

#Создание классификатора и известными параметрами и обучение
lgc = LogisticRegression(penalty="l2", C=c)
lgc.fit(X_train, Y_train)

#Преобработка тестовой выборки
X_test = data_test.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero',\
	"d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], axis=1)

X_test = find_X_train(data_test, X_test)

#Прогноз
prediction = lgc.predict_proba(X_test)
#Искомые величины
maxim = prediction.max()
minim = prediction.min()

print("Max =", maxim, "Min =", minim) #0.9964051844033891 0.0035948155966109008

#Ответ: Max = 0.9964051844033891, Min = 0.0035948155966109008.








