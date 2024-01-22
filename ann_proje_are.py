import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers



# read the csv
df = pd.read_csv('C:/Users/aliek/Desktop/ANN_Paper_Project/heart.csv')

# Define the continuous features
f_continuous = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Identify the features to be converted to object data type
f_convert = list(set(df.columns) - set(f_continuous))

# Convert the identified features to object data type
df[f_convert] = df[f_convert].astype('object')

# Filter out categorical features for the univariate analysis
f_categorical = df.columns.difference(f_continuous)
df_categorical = df[f_categorical]

# Set color palette
sns.set_palette(['#5ffa64', '#fa7a7a'])

import seaborn as sns
import matplotlib.pyplot as plt

# Set up subplots for each continuous feature
fig, ax = plt.subplots(len(f_continuous), 2, figsize=(12, 3*len(f_continuous)))

# Loop through each continuous feature
for i, col in enumerate(f_continuous):
    # Barplot showing the mean value of the feature for each target category
    sns.barplot(data=df, x="target", y=col, ax=ax[i, 0]).set_title(f"{col} Barplot")

    # KDE plot showing the distribution of the feature for each target category
    sns.kdeplot(data=df, x=col, hue="target", fill=True, linewidth=2, ax=ax[i, 1]).set_title(f"{col} KDE Plot")
    ax[i, 1].legend(title='Heart Disease')
    ax[i, 1].set_yticks([])

    # Add mean values to the barplot
    for container in ax[i, 0].containers:
        ax[i, 0].bar_label(container, fmt='%.3g', label_type='edge', fontsize=8)

        
# Set the title for the entire figure
plt.suptitle('Continuous Features vs Target Distribution', fontsize=22)
plt.tight_layout()                     
plt.show()

# Remove 'target' from the f_categorical
f_categorical = [feature for feature in f_categorical if feature != 'target']

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,10))

for i,col in enumerate(f_categorical):
    
    # Create a cross tabulation showing the proportion of purchased and non-purchased loans for each category of the feature
    cross_tab = pd.crosstab(index=df[col], columns=df['target'])
    
    # Using the normalize=True argument gives us the index-wise proportion of the data
    cross_tab_prop = pd.crosstab(index=df[col], columns=df['target'], normalize='index')

    # Define colormap
    cmp = ListedColormap(['#ff826e', 'red'])
    
    # Plot stacked bar charts
    x, y = i//4, i%4
    cross_tab_prop.plot(kind='bar', ax=ax[x,y], stacked=True, width=0.8, colormap=cmp,
                        legend=False, ylabel='Proportion', sharey=True)
    
    # Add the proportions and counts of the individual bars to our plot
    for idx, val in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_location) in zip(cross_tab_prop.loc[val],cross_tab.loc[val],cross_tab_prop.loc[val].cumsum()):
            ax[x,y].text(x=idx-0.3, y=(y_location-proportion)+(proportion/2)-0.03,
                         s = f'    {count}\n({np.round(proportion * 100, 1)}%)', 
                         color = "black", fontsize=9, fontweight="bold")
    
    # Add legend
    ax[x,y].legend(title='target', loc=(0.7,0.9), fontsize=8, ncol=2)
    # Set y limit
    ax[x,y].set_ylim([0,1.12])
    # Rotate xticks
    ax[x,y].set_xticklabels(ax[x,y].get_xticklabels(), rotation=0)
    
            
plt.suptitle('Categorical Features vs Target Stacked Barplots', fontsize=22)
plt.tight_layout()                     
plt.show()

data = df[~df.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)

X = np.array(data.drop(['target'], axis=1))
X_not_age = np.array(data.drop(['age','target'], axis=1))
X_not_sex = np.array(data.drop(['sex','target'], axis=1))
X_not_fbs = np.array(data.drop(['fbs','target'], axis=1))

y = np.array(data['target'])

columns_to_normalize = [X, X_not_age, X_not_sex, X_not_fbs]

for data in columns_to_normalize:
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)

X_train_not_age, X_test_not_age, y_train, y_test = model_selection.train_test_split(X_not_age, y, stratify=y, random_state=42, test_size = 0.2)
X_train_not_sex, X_test_not_sex, y_train, y_test = model_selection.train_test_split(X_not_sex, y, stratify=y, random_state=42, test_size = 0.2)
X_train_not_fbs, X_test_not_fbs, y_train, y_test = model_selection.train_test_split(X_not_fbs, y, stratify=y, random_state=42, test_size = 0.2)

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)

# define a function to build the keras model
def create_model_n():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model_n = create_model_n()

# fit the model to the training data
history=model_n.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10)


# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model_n.predict(X_test), axis=1)

print('Results for Main Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=12, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_model()

# fit the model to the training data
history_not_age=model.fit(X_train_not_age, Y_train, validation_data=(X_test_not_age, Y_test),epochs=50, batch_size=10)

# Model accuracy
plt.plot(history_not_age.history['accuracy'])
plt.plot(history_not_age.history['val_accuracy'])
plt.title('Model Accuracy Without Age')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history_not_age.history['loss'])
plt.plot(history_not_age.history['val_loss'])
plt.title('Model Loss Without Age')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(X_test_not_age), axis=1)

print('Results for without Age Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

history_not_sex=model.fit(X_train_not_sex, Y_train, validation_data=(X_test_not_sex, Y_test),epochs=50, batch_size=10)

# Model accuracy
plt.plot(history_not_sex.history['accuracy'])
plt.plot(history_not_sex.history['val_accuracy'])
plt.title('Model Accuracy Without Sex')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history_not_sex.history['loss'])
plt.plot(history_not_sex.history['val_loss'])
plt.title('Model Loss Without Sex')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(X_test_not_sex), axis=1)

print('Results for without Gender Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

history_not_fbs=model.fit(X_train_not_fbs, Y_train, validation_data=(X_test_not_fbs, Y_test),epochs=50, batch_size=10)

# Model accuracy
plt.plot(history_not_fbs.history['accuracy'])
plt.plot(history_not_fbs.history['val_accuracy'])
plt.title('Model Accuracy Without FBS')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history_not_fbs.history['loss'])
plt.plot(history_not_fbs.history['val_loss'])
plt.title('Model Loss Without FBS')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(X_test_not_fbs), axis=1)

print('Results for without FBS Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))