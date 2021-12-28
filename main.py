from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sci_analysis import analyze

import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")

np.random.seed(123)
stdscaler = StandardScaler()
mmscaler = MinMaxScaler()
data = load_iris()

lr = LogisticRegression()


def histogram(df, title1, rang, qtt):
    plt.hist(df, bins=21)
    plt.title(title1)
    plt.xlabel(rang)
    plt.ylabel(qtt)


def analyzer(lst_name, lst_data, title2):
    analyze({lst_name[0]: lst_data[0], lst_name[1]: lst_data[1], lst_name[2]: lst_data[2], lst_name[3]: lst_data[3]},
            title=title2, nqp=False)


# Raw Data
lst1 = st.norm.rvs(83, 5, size=100).astype(int)
lst2 = st.norm.rvs(74, 6, size=100).astype(int)
lst3 = st.gamma.rvs(1.7, size=100).astype(int)
lst4 = st.gamma.rvs(133, size=100).astype(int)
names = ['lst1', 'lst2', 'lst3', 'lst4']

datas = [lst1, lst2, lst3, lst4]
analyzer(names, datas, 'Raw Data')

titlen = "Normal Distribution"
titleg = "Gamma Distribution"
x = "Range"
y = "Quantity"

histogram(lst1, titlen, x, y)
plt.savefig("images/rd_001.png")
histogram(lst2, titlen, x, y)
plt.savefig("images/rd_002.png")
histogram(lst3, titleg, x, y)
plt.savefig("images/rd_003.png")
histogram(lst4, titleg, x, y)
plt.savefig("images/rd_004.png")

# Case without parameters
stdlst = [lst1, lst2, lst3, lst4]
stdscaler.fit(stdlst)
stdlst = stdscaler.transform(stdlst)

datas = [stdlst[0], stdlst[1], stdlst[2], stdlst[3]]
analyzer(names, datas, 'Scaled Data by Standard Scale with not Parameters')

histogram(stdlst[0], titlen, x, y)
plt.savefig("images/sdss_001.png")
histogram(stdlst[1], titlen, x, y)
plt.savefig("images/sdss_002.png")
histogram(stdlst[2], titleg, x, y)
plt.savefig("images/sdss_003.png")
histogram(stdlst[3], titleg, x, y)
plt.savefig("images/sdss_004.png")

# Case with parameters
stdlst = {'list1': lst1, 'list2': lst2, 'list3': lst3, 'list4': lst4}
df_stdlst = pd.DataFrame(stdlst)
list1scl = stdscaler.fit_transform(df_stdlst['list1'].values.reshape(-1, 1))
list2scl = stdscaler.fit_transform(df_stdlst['list2'].values.reshape(-1, 1))
list3scl = stdscaler.fit_transform(df_stdlst['list3'].values.reshape(-1, 1))
list4scl = stdscaler.fit_transform(df_stdlst['list4'].values.reshape(-1, 1))

datas = [list1scl, list2scl, list3scl, list4scl]
analyzer(names, datas, 'Scaled Data by Standard Scale with Parameters')

histogram(list1scl, titlen, x, y)
plt.savefig("images/sdssp_001.png")
histogram(list2scl, titlen, x, y)
plt.savefig("images/sdssp_002.png")
histogram(list3scl, titleg, x, y)
plt.savefig("images/sdssp_003.png")
histogram(list4scl, titleg, x, y)
plt.savefig("images/sdssp_004.png")

# Min Max Scaled Data
mmlst = [lst1, lst2, lst3, lst4]
mmscaler.fit(mmlst)
mmlst = mmscaler.transform(mmlst)

datas = [mmlst[0], mmlst[1], mmlst[2], mmlst[3]]
analyzer(names, datas, 'Scaled Data by Min Max Scaled')

histogram(mmlst[0], titlen, x, y)
plt.savefig("images/sdmms_001.png")
histogram(mmlst[1], titlen, x, y)
plt.savefig("images/sdmms_002.png")
histogram(mmlst[2], titleg, x, y)
plt.savefig("images/sdmms_003.png")
histogram(mmlst[3], titleg, x, y)    # plot blank
plt.savefig("images/sdmms_004.png")

# Residual Data
lst1_resi = lst1 - int(np.mean(lst1))
lst2_resi = lst2 - int(np.mean(lst2))
lst3_resi = lst3 - int(np.mean(lst3))
lst4_resi = lst4 - int(np.mean(lst4))

datas = [lst1_resi, lst2_resi, lst3_resi, lst4_resi]
analyzer(names, datas, 'Scaled Data by Residual Values')

histogram(lst1_resi, titlen, x, y)
plt.savefig("images/sdrv_001.png")
histogram(lst2_resi, titlen, x, y)
plt.savefig("images/sdrv_002.png")
histogram(lst3_resi, titleg, x, y)
plt.savefig("images/sdrv_003.png")
histogram(lst4_resi, titleg, x, y)
plt.savefig("images/sdrv_004.png")

# Scaled Data by Equation
max1 = np.max(lst1)
min1 = np.min(lst1)
max2 = np.max(lst2)
min2 = np.min(lst2)
max3 = np.max(lst3)
min3 = np.min(lst3)
max4 = np.max(lst4)
min4 = np.min(lst4)

lst1_scaled = np.array([(x - min1)/(max1 - min1) for x in lst1])
lst2_scaled = np.array([(x - min2)/(max2 - min2) for x in lst2])
lst3_scaled = np.array([(x - min3)/(max3 - min3) for x in lst3])
lst4_scaled = np.array([(x - min4)/(max4 - min4) for x in lst4])

datas = [lst1_scaled, lst2_scaled, lst3_scaled, lst4_scaled]
analyzer(names, datas, 'Scaled Data by Equation')

histogram(lst1_scaled, titlen, x, y)
plt.savefig("images/sde_001.png")
histogram(lst2_scaled, titlen, x, y)
plt.savefig("images/sde_002.png")
histogram(lst3_scaled, titleg, x, y)
plt.savefig("images/sde_003.png")
histogram(lst4_scaled, titleg, x, y)
plt.savefig("images/sde_004.png")

# Iris Dataset
datas = [data.data[:, 0], data.data[:, 1], data.data[:, 2], data.data[:, 3]]
names = ['sl', 'sw', 'pl', 'pw']
analyzer(names, datas, 'Iris Dataset')

stdiris = stdscaler.fit_transform(data.data)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=42)
lr.fit(X_train, y_train)
print("Raw Results: {} in train and {} in test.".format(lr.score(X_train, y_train), lr.score(X_test, y_test)))

Xs_train, Xs_test, ys_train, ys_test = train_test_split(stdiris, data.target, test_size=0.30, random_state=42)
lr.fit(Xs_train, ys_train)
print("Scaled Results: {} in train and {} in test.".format(lr.score(Xs_train, ys_train), lr.score(Xs_test, ys_test)))

max1 = np.max(data.data[:, 0])
min1 = np.min(data.data[:, 0])
max2 = np.max(data.data[:, 1])
min2 = np.min(data.data[:, 1])
max3 = np.max(data.data[:, 2])
min3 = np.min(data.data[:, 2])
max4 = np.max(data.data[:, 3])
min4 = np.max(data.data[:, 3])

iris1_scaled = np.array([(x - min1)/(max1 - min1) for x in data.data[:, 0]])
iris2_scaled = np.array([(x - min2)/(max2 - min2) for x in data.data[:, 1]])
iris3_scaled = np.array([(x - min3)/(max3 - min3) for x in data.data[:, 2]])
# iris4_scaled = np.array([(x - min4)/(max4 - min4) for x in data.data[:, 3]])

iris = pd.DataFrame({'sl': iris1_scaled, 'sw': iris2_scaled, 'pl': iris3_scaled})

X_trains, X_tests, y_trains, y_tests = train_test_split(iris, data.target, test_size=0.30, random_state=42)
lr.fit(X_trains, y_trains)
print("Equation Results: {} in train and {} in test.".format(lr.score(X_trains, y_trains), lr.score(X_tests, y_tests)))
