# Market-Basket-Analysis
# Market Basket Analysis enhances retail by uncovering customer buying patterns, optimizing inventory, and boosting profitability.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Groceries_dataset.csv")
df.head()
df.info()
df.isnull().sum().sort_values(ascending=False)
df['date']=pd.to_datetime(df["Date"])
df.info()
item_distr=df.groupby(by="itemDescription").size().reset_index(name="frequency").sort_values(by="frequency",ascending=False).head(10)

bars=item_distr["itemDescription"]
height=item_distr["frequency"]
x_pos=np.arange(len(bars))

plt.figure(figsize=(16,9))
plt.bar(x_pos,height,color=(0.2,0.3,0.5,0.5))
plt.title("top 10 sold items")
plt.xlabel("item names")
plt.ylabel("Number of quantity sold")

plt.xticks(x_pos,bars)

plt.show()
df_date=df.set_index((['Date']))
df_date.head()
df_date['Date'] = pd.to_datetime(df_date['date'])
df_date.set_index('date', inplace=True)
df_date.resample("M")['itemDescription'].count().plot(figsize=(20, 8), grid=True, title="Number of items sold by month")
plt.xlabel("Date")
plt.ylabel("Number of items sold")
plt.show()
cust_level=df[["Member_number","itemDescription"]].sort_values(by="Member_number",ascending=False)
cust_level["itemDescription"]=cust_level["itemDescription"].str.strip()
cust_level
transactions=[a[1]['itemDescription'].tolist() for a in list(cust_level.groupby("Member_number"))]
transactions
from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.002,min_confidence=0.05,min_lift=3,min_length=2)
result=list(rules)
def inspects(result):
    lhs=[tuple(results[2][0][0]) [0] for results in result]
    rhs=[tuple(results[2][0][1]) [0] for results in result]
    supports=[results[1] for results in result ]
    confidence=[results[2][0][2] for results in result]
    lifts=[results[2][0][3] for results in result]
    return list(zip(lhs,rhs,supports,confidence,lifts))

resultsdataframe=pd.DataFrame(inspects(result),columns=['Left Hand Side','Right Hand Side', 'Support', 'Confidences','Lifts'])

resultsdataframe.nlargest(n=10,columns='Lifts')
