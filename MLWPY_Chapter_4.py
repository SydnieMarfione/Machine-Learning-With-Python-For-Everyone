#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


from sklearn import (datasets,
                     linear_model,
                     metrics,
                     model_selection as skms,
                     neighbors)


# In[15]:


import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# In[16]:


pd.options.display.float_format = '{:20,.4f}'.format


# In[17]:


diabetes = datasets.load_diabetes()

tts = skms.train_test_split(diabetes.data,
                            diabetes.target,
                            test_size=.25)

(diabetes_train_ftrs, diabetes_test_ftrs,
diabetes_train_tgt,  diabetes_test_tgt) = tts


# In[18]:


diabetes_df = pd.DataFrame(diabetes.data,
                           columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target
diabetes_df.head()


# In[19]:


sns.pairplot(diabetes_df[['age', 'sex', 'bmi', 'bp', 's1']],
             height=1.5, hue='sex', plot_kws={'alpha':.2});


# In[21]:


values = np.array([1, 3, 5, 8, 11, 13, 15])
print("no outlier")
print(np.mean(values), np.median(values))

values_with__outlier = np.array([1, 3, 5, 8, 11, 13, 40])
print("with outlier")
print("%5.2f" % np.mean(values_with__outlier), np.median(values_with__outlier))


# In[22]:


distances = np.array([2.0, 4.0, 4.0])
closeness = 1.0 / distances
weights = closeness / np.sum(closeness)
weights


# In[23]:


values = np.array([4, 6, 8])

mean = np.mean(values)
wgt_mean = np.dot(values, weights)

print("Mean:", mean)
print("Weight Mean:", wgt_mean)


# In[24]:


knn = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = knn.fit(diabetes_train_ftrs, diabetes_train_tgt)
preds = fit.predict(diabetes_test_ftrs)

metrics.mean_squared_error(diabetes_test_tgt, preds)


# In[25]:


np.sqrt(3500)


# In[26]:


diabetes_df["target"].max() - diabetes_df["target"].min()


# In[27]:


def axis_helper(ax, lims):

    ax.set_xlim(lims); ax.set_xticks([])
    ax.set_ylim(lims); ax.set_yticks([])
    ax.set_aspect('equal')


# In[28]:


D = np.array([[3, 5],
             [4,2]])

x,y = D[:,0], D[:,1]


# In[30]:


horizontal_lines = np.array([1, 2, 3, 3.5, 4, 5])

results = []
fig, axes = plt.subplots(1, 6, figsize=(10, 5))
for h_line, ax in zip(horizontal_lines, axes.flat):
    axis_helper(ax, (0, 6))
    ax.set_title(str(h_line))
    
    ax.plot(x, y, 'ro')
    
    ax.axhline(h_line, color='y')
    
    predictions = h_line
    ax.vlines(x, predictions, y)
    
    errors = y - predictions
    sse = np.dot(errors, errors)
    
    results.append((predictions, errors, errors.sum(), sse, np.sqrt(sse)))


# In[31]:


col_labels = "Prediction", "Errors", "Sum", "SSE", "Distance"
display(pd.DataFrame.from_records(results,
                                 columns=col_labels,
                                 index="Prediction"))


# In[32]:


def process(D, model, ax):
    x,y = D[:,0], D[:,1]
    m,b = model   
    
    axis_helper(ax, (0,8))
    
    ax.plot(x,y,'ro')

    helper_xs   = np.array([0,8])
    helper_line = m * helper_xs + b 
    ax.plot(helper_xs, helper_line, color='y')
    
    predictions = m * x + b
    ax.vlines(x, predictions, y)
    
    errors = y - predictions

    sse = np.dot(errors, errors) 
    return (errors, errors.sum(), sse, np.sqrt(sse))


# In[35]:


D = np.array([[3, 5],
              [4, 2]])

lines_mb = np.array([[1, 0],
                    [1, 1],
                    [1, 2],
                    [-1, 8],
                    [-3, 14]])

col_labels = ("Raw Errors", "Sum", "SSE", "TotDist")
results = []

fig, axes = plt.subplots(1, 5, figsize=(12, 6))
records = [process(D, mod, ax) for mod, ax in zip(lines_mb, axes.flat)]
df = pd.DataFrame.from_records(records, columns=col_labels)
display(df)


# In[36]:


lr = linear_model.LinearRegression()
fit = lr.fit(diabetes_train_ftrs, diabetes_train_tgt)
preds = fit.predict(diabetes_test_ftrs)

metrics.mean_squared_error(diabetes_test_tgt, preds)


# In[37]:


tgt = np.array([3, 5, 8, 10, 12, 15])


# In[38]:


num_guesses = 10
results = []

for g in range(num_guesses):
    guess = np.random.uniform(low=tgt.min(), high=tgt.max())
    total_dist = np.sum((tgt - guess)**2)
    results.append((total_dist, guess))
best_guess = sorted(results)[0][1]
best_guess


# In[39]:


num_steps = 100
step_size = .05

best_guess = np.random.uniform(low=tgt.min(), high=tgt.max())
best_dist = np.sum((tgt - best_guess)**2)

for s in range(num_steps):
    new_guess = best_guess + (np.random.choice([+1, -1]) * step_size)
    new_dist = np.sum((tgt - new_guess)**2)
    if new_dist < best_dist:
        best_guess, best_dist = new_guess, new_dist
print(best_guess)


# In[41]:


num_steps = 1000
step_size = 0.02
best_guess = np.random.uniform(low=tgt.min(), high=tgt.max())
best_dist = np.sum((tgt - best_guess) ** 2)
print("start:", best_guess)

for s in range(num_steps):
    guesses = best_guess + (np.array([-1, 1]) * step_size)
    dists = np.sum((tgt[:, np.newaxis] - guesses) ** 2, axis=0)

    better_idx = np.argmin(dists)
    if dists[better_idx] > best_dist:
        break

    best_guess = guesses[better_idx]
    best_dist = dists[better_idx]

print(" end:", best_guess)


# In[42]:


print("mean:", np.mean(tgt))


# In[44]:


diabetes = datasets.load_diabetes()
tts = skms.train_test_split(diabetes.data,
                            diabetes.target,
                            test_size=.25)

(diabetes_train_ftrs, diabetes_test_ftrs,
 diabetes_train_tgt, diabetes_test_tgt) = tts

models = {'kNN': neighbors.KNeighborsRegressor(n_neighbors=3),
          'linreg': linear_model.LinearRegression()}

for name, model in models.items():
    fit = model.fit(diabetes_train_ftrs, diabetes_train_tgt)
    preds = fit.predict(diabetes_test_ftrs)

    score = np.sqrt(metrics.mean_squared_error(diabetes_test_tgt, preds))
    print("{:>6s} : {:.2f}".format(name, score))

